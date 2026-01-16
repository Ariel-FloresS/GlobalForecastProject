from .segmented_forecast_orchestrator_interface import SegmentedForecastOrchestatorInterface
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelectorInterface
from model.business_layer.forecasting.model_factory import ModelFactoryInterface, DistributedModel
from model.business_layer.forecasting.forecasting_engine import DistributedForecastingEngineInterface, DistributedForecastingEngine
from model.business_layer.forecasting.config import ModelSpec
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
import pyspark.sql.functions as F
from typing import Optional, List


class SegmentedForecastOrchestator(SegmentedForecastOrchestatorInterface):

    def __init__(self,
                classification: str,
                cluster_spec_selector: ClusterSpecSelectorInterface,
                model_factory: ModelFactoryInterface)->None:
        
        
        self.classification = classification
        self.cluster_spec_selector = cluster_spec_selector
        self.model_factory = model_factory
        


    def forecast(self, training_dataset: DataFrame, future_dataset: DataFrame, frequency:str,horizon: int, static_features:Optional[List[str]] = None)->DataFrame:


        classification_training_dataset: DataFrame = training_dataset.filter(F.col('classification')== self.classification).drop('classification')

        if classification_training_dataset.rdd.isEmpty():

            empty_predictions: DataFrame = (
                classification_training_dataset
                .select('unique_id', 'ds')
                .withColumn('y_pred', F.lit(None).cast('double'))
                .limit(0)
            )

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
        if not static_features:

            static_features: List[None] = []
        
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


        return predictions_dataframe





        




        
        


        