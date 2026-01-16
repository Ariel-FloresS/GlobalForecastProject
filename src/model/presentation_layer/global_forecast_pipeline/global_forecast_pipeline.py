from .global_forecast_pipeline_interface import GlobalForecastPipelineInterface
from model.data_layer.repositories.training_data import  TrainingDataRepository
from model.data_layer.repositories.forecast_data import ForecastDataRepositoryInterface, ForecastDataRepository
from model.data_layer.dataset_partitioning import  DatasetPartitioning
from model.business_layer.data_preparation import DataPreparationInterface, DataPreparation
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelector
from model.business_layer.forecasting.model_factory import ModelFactory
from model.business_layer.segmented_forecast_orchestrator import SegmentedForecastOrchestatorInterface, SegmentedForecastOrchestator
from feature_store.presentation_layer import  FeatureStore
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional


class GlobalForecastPipeline(GlobalForecastPipelineInterface):

    def __init__(self, spark:SparkSession, training_delta_table:str, exogenous_columns:List[str], forecast_data_delta_table:str)->None:
        self.spark = spark
        self.training_delta_table = training_delta_table
        self.exogenous_columns = exogenous_columns
        self.forecast_data_delta_table = forecast_data_delta_table

    def forecast(self ,frequency:str, season_lenght:int, horizon:int, static_features:Optional[List[str]] = None)->None:

        forecast_data_repository: ForecastDataRepositoryInterface = ForecastDataRepository()
        
        data_preparation_pipeline: DataPreparationInterface = DataPreparation(training_repository = TrainingDataRepository(spark = self.spark),
                                                                    dataset_partitioning = DatasetPartitioning(spark = self.spark),
                                                                    feature_store = FeatureStore(spark = self.spark, frequency = frequency, season_length = season_lenght)
                                                                    )
        
        
        training_dataset, future_dataset = data_preparation_pipeline.prepare_batch_training_and_future_datasets(training_delta_table = self.training_delta_table,
                                                                                                                exogenous_columns = self.exogenous_columns,
                                                                                                                horizon = horizon)
        
        smooth_segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = 'Smooth', 
                                                                                                        cluster_spec_selector = ClusterSpecSelector(),
                                                                                                        model_factory = ModelFactory())
        
        erratic_segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = 'Erratic',
                                                                                                        cluster_spec_selector = ClusterSpecSelector(),
                                                                                                        model_factory = ModelFactory()
                                                                                                        )
        
        lumpy_segmente_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = 'Lumpy',
                                                                                                    cluster_spec_selector = ClusterSpecSelector(),
                                                                                                    model_factory = ModelFactory()
                                                                                                      )
        
        intermittent_segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification =  'Intermittent',
                                                                                                              cluster_spec_selector = ClusterSpecSelector(),
                                                                                                                model_factory = ModelFactory()
                                                                                                              )
        
        smooth_predictions: DataFrame = smooth_segmented_forecast.forecast(training_dataset = training_dataset,
                                                                            future_dataset = future_dataset, 
                                                                            frequency = frequency,
                                                                            horizon = horizon,
                                                                            static_features = static_features)
        
        erratic_predictions: DataFrame = erratic_segmented_forecast.forecast(training_dataset = training_dataset,
                                                                             future_dataset = future_dataset,
                                                                             frequency = frequency,
                                                                             horizon = horizon,
                                                                             static_features = static_features)
        
        
        lumpy_predictions: DataFrame = lumpy_segmente_forecast.forecast(training_dataset = training_dataset,
                                                                        future_dataset = future_dataset,
                                                                        frequency = frequency,
                                                                        horizon =  horizon,
                                                                        static_features = static_features)
        
        intermittent_predictions: DataFrame = intermittent_segmented_forecast.forecast(training_dataset = training_dataset,
                                                                                       future_dataset = future_dataset,
                                                                                       frequency = frequency,
                                                                                       horizon = horizon,
                                                                                       static_features = static_features)
        
        prediction_dataframe_output: DataFrame = (smooth_predictions
                                                  .unionByName(erratic_predictions)
                                                  .unionByName(lumpy_predictions)
                                                  .unionByName(intermittent_predictions)
                                                  )
        
        forecast_data_repository.save_forecast_data(forecast_dataframe = prediction_dataframe_output,
                                                     delta = self.forecast_data_delta_table)
    

        



    
        