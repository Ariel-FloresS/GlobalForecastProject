from .segmented_forecast_orchestrator_interface import SegmentedForecastOrchestatorInterface
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelectorInterface
from model.business_layer.forecasting.model_factory import ModelFactoryInterface, DistributedModel, LocalModel
from model.business_layer.forecasting.forecasting_engine import DistributedForecastingEngineInterface, DistributedForecastingEngine
from model.business_layer.forecasting.config import ModelSpec
from model.data_layer.dtos import ArtefactSpec
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
import pyspark.sql.functions as F
from typing import Optional, List, Dict, Any
from loguru import logger



class SegmentedForecastOrchestator(SegmentedForecastOrchestatorInterface):

    def __init__(self,
                classification: str,
                cluster_spec_selector: ClusterSpecSelectorInterface,
                model_factory: ModelFactoryInterface)->None:
        
        
        self.classification = classification
        self.cluster_spec_selector = cluster_spec_selector
        self.model_factory = model_factory
        


    def forecast(self, training_dataset: DataFrame, future_dataset: DataFrame, frequency:str,horizon: int, static_features:Optional[List[str]] = None)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"[{step_name}] Start Forecast for the classification: {self.classification}.\n")

        classification_training_dataset: DataFrame = training_dataset.filter(F.col('classification')== self.classification).drop('classification')

        if classification_training_dataset.rdd.isEmpty():

            empty_predictions: DataFrame = (
                classification_training_dataset
                .select('unique_id', 'ds')
                .withColumn('y_pred', F.lit(None).cast('double'))
                .limit(0)
            )

            logger.info(f"[{step_name}] There is no Training Data for the classification '{self.classification}' Skipping The Training Process.\n")

            return empty_predictions

        unique_ids_classification: DataFrame = classification_training_dataset.select('unique_id').distinct()

        future_dataset_classification: DataFrame = (future_dataset 
                                                    .join(other = unique_ids_classification, on = 'unique_id', how = 'inner')
                                                    )
        
        model_spec: ModelSpec = self.cluster_spec_selector.get_spec_by_classification(classification = self.classification)

        models: List[DistributedModel] = self.model_factory.built_models(models_config = model_spec.models)

        distributed_forecast_engine: DistributedForecastingEngineInterface = DistributedForecastingEngine(models = models,
                                                                                                          frequency = frequency,
                                                                                                          lags = model_spec.lags,
                                                                                                          lag_transforms = model_spec.lag_transforms,
                                                                                                          target_transforms = model_spec.target_transforms)
        static_features: List[str] = static_features or []
        
        distributed_forecast_engine.fit(training_dataset = classification_training_dataset, static_features = static_features)

        predictions_dataframe: DataFrame = distributed_forecast_engine.predict(prediction_horizon = horizon, future_dataframe = future_dataset_classification)

        prediction_columns: List[str] =[ col for col in predictions_dataframe.columns if col not in ['unique_id', 'ds'] ]


        # When only one model is used, its output is taken as the final forecast.
        # When multiple models are used, their predictions are combined using a mean ensemble.
        if len(prediction_columns)==1:

            predictions_dataframe: DataFrame = (predictions_dataframe
                                                .withColumnRenamed(prediction_columns[0], 'y_pred')
                                                .select('unique_id', 'ds', 'y_pred')
                                                )
        else:

            sum_expr: Column = sum( F.coalesce(F.col(c), F.lit(0.0)) for c in prediction_columns )
            count_expr: Column = sum( F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0)) for c in prediction_columns )

            predictions_dataframe: DataFrame = (predictions_dataframe
                                                .withColumn('y_pred', 
                                                            F.when(count_expr > 0, sum_expr / count_expr)
                                                            .otherwise(F.lit(None).cast("double"))
                                                            )
                                                .select('unique_id', 'ds', 'y_pred')
                                                )

        logger.info(f"[{step_name}] End Forecast for the classification: {self.classification}.\n")
        return predictions_dataframe


    def cross_validation(self,
                        training_dataset:DataFrame,
                        frequency:str,
                        windows:int,
                        periods_for_each_window:int,
                        static_features:Optional[List[str]] = None)->DataFrame:
        
        step_name: str = self.__class__.__name__

        logger.info(f"[{step_name}] Start Cross Validation For the Classification: {self.classification}.\n")

        classification_training_dataset: DataFrame = training_dataset.filter(F.col('classification')== self.classification).drop('classification')

        if classification_training_dataset.rdd.isEmpty():

            empty_predictions: DataFrame = (
                classification_training_dataset
                .select('unique_id', 'ds')
                .withColumn('y_pred', F.lit(None).cast('double'))
                .withColumn('classification', F.lit(None).cast('string'))
                .withColumn('cutoff', F.lit(None).cast('string'))
                .limit(0)
            )

            logger.info(f"[{step_name}] There is no Training Data for the classification '{self.classification}' Skipping The Training Process.\n")

            return empty_predictions
        
        model_spec: ModelSpec = self.cluster_spec_selector.get_spec_by_classification(classification = self.classification)

        models: List[DistributedModel] = self.model_factory.built_models(models_config = model_spec.models)

        distributed_forecast_engine: DistributedForecastingEngineInterface = DistributedForecastingEngine(models = models,
                                                                                                          frequency = frequency,
                                                                                                          lags = model_spec.lags,
                                                                                                          lag_transforms = model_spec.lag_transforms,
                                                                                                          target_transforms = model_spec.target_transforms)
        static_features: List[str] = static_features or []

        cross_validation_dataframe: DataFrame = distributed_forecast_engine.cross_validation(training_dataset = classification_training_dataset,
                                                                                            windows = windows,
                                                                                            periods_for_each_window = periods_for_each_window,
                                                                                            static_features = static_features)
        
        prediction_column: List[str] =[ col for col in cross_validation_dataframe.columns if col not in ['unique_id', 'ds', 'cutoff', 'y'] ]

        cross_validation_dataframe: DataFrame = (cross_validation_dataframe
                                                .withColumnRenamed(prediction_column[0], 'y_pred')
                                                .withColumn('classification', self.classification)
                                                )
        return cross_validation_dataframe
        

    def train_and_get_local_model(self,
                    training_dataset: DataFrame,
                    frequency:str,
                    static_features:Optional[List[str]] = None)->ArtefactSpec:
        
        step_name: str = self.__class__.__name__

        logger.info(f"[{step_name}] Start Training and get the local model for the classification: {self.classification}.\n")

        classification_training_dataset: DataFrame = training_dataset.filter(F.col('classification')== self.classification).drop('classification')

        if classification_training_dataset.rdd.isEmpty():

            logger.info(f"[{step_name}] There is no Training Data for the classification '{self.classification}' Skipping The Training Process And Returning None.\n")

            return None
        
        model_spec: ModelSpec = self.cluster_spec_selector.get_spec_by_classification(classification = self.classification)

        models: List[DistributedModel] = self.model_factory.built_models(models_config = model_spec.models)

        static_features: List[str] = static_features or []

        distributed_forecast_engine: DistributedForecastingEngineInterface = DistributedForecastingEngine(models = models,
                                                                                                          frequency = frequency,
                                                                                                          lags = model_spec.lags,
                                                                                                          lag_transforms = model_spec.lag_transforms,
                                                                                                          target_transforms = model_spec.target_transforms)
        static_features: List[str] = static_features or []
        
        distributed_forecast_engine.fit(training_dataset = classification_training_dataset, static_features = static_features)

        feature_columns: List[str] = (distributed_forecast_engine
                                      .preprocess(training_dataset = classification_training_dataset)
                                      .drop('unique_id', 'ds', 'y').columns
                                    )

        train_distributed_models: Dict[DistributedModel] = distributed_forecast_engine.distributed_ml_forecast.models_

        model_name, local_model = self.model_factory.built_local_model(distributed_model = train_distributed_models)

        return ArtefactSpec(model_name =model_name,
                            classification = self.classification,
                            local_model = local_model,
                            features_columns = feature_columns)

        
        
        


        
    