from .global_forecast_pipeline_interface import GlobalForecastPipelineInterface
from model.data_layer.repositories.training_data import  TrainingDataRepository
from model.data_layer.repositories.forecast_data import ForecastDataRepositoryInterface, ForecastDataRepository
from model.data_layer.dataset_partitioning import  DatasetPartitioning, DatasetPartitioningInterface
from model.business_layer.data_preparation import DataPreparationInterface, DataPreparation
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelector
from model.business_layer.forecasting.model_factory import ModelFactory
from model.business_layer.segmented_forecast_orchestrator import SegmentedForecastOrchestatorInterface, SegmentedForecastOrchestator
from feature_store.presentation_layer import  FeatureStore
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional
from loguru import logger
import pyspark.sql.functions as F

class GlobalForecastPipeline(GlobalForecastPipelineInterface):

    def __init__(self, spark:SparkSession, training_delta_table:str, exogenous_columns:List[str], forecast_data_delta_table:str)->None:
        self.spark = spark
        self.training_delta_table = training_delta_table
        self.exogenous_columns = exogenous_columns
        self.forecast_data_delta_table = forecast_data_delta_table


    def _forecast_one_classification(self,
                                    training_dataset: DataFrame, 
                                    future_dataset:DataFrame,
                                    classification: str,
                                    frequency:str, horizon:int,
                                    static_features:Optional[List[str]] = None)->DataFrame:
       
       segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = classification, 
                                                                                                cluster_spec_selector = ClusterSpecSelector(),
                                                                                                model_factory = ModelFactory())
       return segmented_forecast.forecast(training_dataset = training_dataset,
                                          future_dataset = future_dataset, 
                                          frequency = frequency,
                                          horizon = horizon,
                                          static_features = static_features)
       
       

       

    def forecast(self ,frequency:str, season_lenght:int, horizon:int, version:str ,static_features:Optional[List[str]] = None)->None:
        
        pipeline_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[FORECAST PIPELINE START]   {pipeline_name}\n{'='*84}"
        logger.info(banner_top)

        forecast_data_repository: ForecastDataRepositoryInterface = ForecastDataRepository()
        
        data_preparation_pipeline: DataPreparationInterface = DataPreparation(training_repository = TrainingDataRepository(spark = self.spark),
                                                                    dataset_partitioning = DatasetPartitioning(spark = self.spark),
                                                                    feature_store = FeatureStore(spark = self.spark, frequency = frequency, season_length = season_lenght)
                                                                    )
        
        
        training_dataset, future_dataset = data_preparation_pipeline.prepare_batch_training_and_future_datasets(training_delta_table = self.training_delta_table,
                                                                                                                exogenous_columns = self.exogenous_columns,
                                                                                                                horizon = horizon,
                                                                                                                static_features = static_features)
      
        
        smooth_predictions: DataFrame = self._forecast_one_classification(training_dataset = training_dataset,
                                                                        future_dataset = future_dataset,
                                                                        classification = 'Smooth',
                                                                        frequency = frequency,
                                                                        horizon = horizon,
                                                                        static_features = static_features)
        
        erratic_predictions: DataFrame = self._forecast_one_classification(training_dataset = training_dataset,
                                                                        future_dataset = future_dataset,
                                                                        classification = 'Erratic',
                                                                        frequency = frequency,
                                                                        horizon = horizon,
                                                                        static_features = static_features)
        
        
        lumpy_predictions: DataFrame = self._forecast_one_classification(training_dataset = training_dataset,
                                                                        future_dataset = future_dataset,
                                                                        classification = 'Lumpy',
                                                                        frequency = frequency,
                                                                        horizon = horizon,
                                                                        static_features = static_features)
        
        intermittent_predictions: DataFrame = self._forecast_one_classification(training_dataset = training_dataset,
                                                                        future_dataset = future_dataset,
                                                                        classification = 'Intermittent',
                                                                        frequency = frequency,
                                                                        horizon = horizon,
                                                                        static_features = static_features)
        
        prediction_dataframe_output: DataFrame = (smooth_predictions
                                                  .unionByName(erratic_predictions)
                                                  .unionByName(lumpy_predictions)
                                                  .unionByName(intermittent_predictions)
                                                  )
        
        forecast_data_repository.save_forecast_data(forecast_dataframe = prediction_dataframe_output,
                                                     delta = self.forecast_data_delta_table,
                                                     version = version)
        
        banner_end: str = f"\n{'='*84}\n[FORECAST PIPELINE END]   {pipeline_name}\n{'='*84}"
        logger.info(banner_end)
    
    def _backtesting_one_classification(
                    self,
                    training_dataset:DataFrame,
                    frequency:str,
                    classification:str,
                    windows:int,
                    periods_for_each_window:int,
                    static_features:Optional[List[str]] = None)->DataFrame:
      
      dataset_partitioning: DatasetPartitioningInterface =  DatasetPartitioning(spark = self.spark)

      training_dataset_partition: DataFrame = dataset_partitioning.get_dataset_partition(dataset = training_dataset,
                                                                                            partition_column = 'unique_id')
        
      segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = classification, 
                                                                                              cluster_spec_selector = ClusterSpecSelector(),
                                                                                              model_factory = ModelFactory())
                                                                                              
      return segmented_forecast.cross_validation(training_dataset = training_dataset_partition,
                                          frequency = frequency,
                                          windows = windows,
                                          periods_for_each_window = periods_for_each_window,
                                          static_features = static_features)
        
    def backtesting(
                    self,
                    training_dataset:DataFrame,
                    frequency:str,
                    windows:int,
                    periods_for_each_window:int,
                    static_features:Optional[List[str]] = None)->DataFrame:
      
      pipeline_name: str = self.__class__.__name__

      banner_top: str = f"\n{'='*84}\n[BACKTESTING PIPELINE START]   {pipeline_name}\n{'='*84}"
      logger.info(banner_top)
        
      smooth_cross_validation_dataframe: DataFrame = self._backtesting_one_classification(training_dataset = training_dataset,
                                                                                          frequency = frequency,
                                                                                          classification = 'Smooth',
                                                                                          windows = windows,
                                                                                          periods_for_each_window = periods_for_each_window,
                                                                                          static_features = static_features)
      
      erratic_cross_validation_dataframe: DataFrame = self._backtesting_one_classification(training_dataset = training_dataset,
                                                                                          frequency = frequency,
                                                                                          classification = 'Erratic',
                                                                                          windows = windows,
                                                                                          periods_for_each_window = periods_for_each_window,
                                                                                          static_features = static_features)
      
      lumpy_cross_validation_dataframe: DataFrame = self._backtesting_one_classification(training_dataset = training_dataset,
                                                                                          frequency = frequency,
                                                                                          classification = 'Lumpy',
                                                                                          windows = windows,
                                                                                          periods_for_each_window = periods_for_each_window,
                                                                                          static_features = static_features)
      
      intermittent_cross_validation_dataframe: DataFrame = self._backtesting_one_classification(training_dataset = training_dataset,
                                                                                          frequency = frequency,
                                                                                          classification = 'Intermittent',
                                                                                          windows = windows,
                                                                                          periods_for_each_window = periods_for_each_window,
                                                                                          static_features = static_features)
      backtesting_dataframe_output: DataFrame = (smooth_cross_validation_dataframe
                                                  .unionByName(erratic_cross_validation_dataframe)
                                                  .unionByName(lumpy_cross_validation_dataframe)
                                                  .unionByName(intermittent_cross_validation_dataframe)
                                                  )
      
      banner_end: str = f"\n{'='*84}\n[BACKTESTING PIPELINE END]   {pipeline_name}\n{'='*84}"
      logger.info(banner_end)
      
      return backtesting_dataframe_output
      
      

        
      
      
        

        



    
        