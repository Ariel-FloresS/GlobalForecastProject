from .segmented_forecast_orchestrator_interface import SegmentedForecastOrchestatorInterface
from forecasting.cluster_spec_selector import ClusterSpecSelectorInterface
from forecasting.model_factory import ModelFactoryInterface, DistributedModel
from forecasting.forecasting_engine import DistributedForecastingEngineInterface, DistributedForecastingEngine
from forecasting.config import ModelSpec
from pyspark.sql import DataFrame
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


        classification_training_dataset: DataFrame = training_dataset.filter(F.col('classification')== self.classification)

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
        
        distributed_forecast_engine.fit(traing_dataset = classification_training_dataset, static_features = static_features)

        predictions_dataframe: DataFrame = distributed_forecast_engine.predict(prediction_horizon = horizon, future_dataframe = future_dataset_classification)

        return predictions_dataframe





        




        
        


        