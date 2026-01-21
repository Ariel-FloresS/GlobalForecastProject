from .global_forecast_pipeline_interface import GlobalForecastPipelineInterface
from model.data_layer.repositories.training_data import  TrainingDataRepository
from model.data_layer.repositories.forecast_data import ForecastDataRepositoryInterface, ForecastDataRepository
from model.data_layer.dataset_partitioning import  DatasetPartitioning, DatasetPartitioningInterface
from model.business_layer.data_preparation import DataPreparationInterface, DataPreparation
from model.business_layer.forecasting.cluster_spec_selector import ClusterSpecSelector
from model.business_layer.forecasting.model_factory import ModelFactory
from model.business_layer.segmented_forecast_orchestrator import SegmentedForecastOrchestatorInterface, SegmentedForecastOrchestator
from model.business_layer.forecasting.model_factory import  LocalModel
from model.data_layer.dtos import ArtefactSpec
from feature_store.presentation_layer import  FeatureStore
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional, Dict, Any
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
    

    def _train_and_get_local_model_one_classification(self,
                                                      training_dataset: DataFrame,
                                                      classification:str,
                                                      frequency:str,
                                                      static_features:Optional[List[str]] = None )->ArtefactSpec:
      
      dataset_partitioning: DatasetPartitioningInterface =  DatasetPartitioning(spark = self.spark)

      training_dataset_partition: DataFrame = dataset_partitioning.get_dataset_partition(dataset = training_dataset,
                                                                                            partition_column = 'unique_id')
      
      segmented_forecast: SegmentedForecastOrchestatorInterface = SegmentedForecastOrchestator(classification = classification, 
                                                                                              cluster_spec_selector = ClusterSpecSelector(),
                                                                                              model_factory = ModelFactory())
      
      return segmented_forecast.train_and_get_local_model(training_dataset = training_dataset_partition,
                                                          frequency = frequency,
                                                          static_features = static_features)
    
    def _get_feature_importance_one_classification(self,
                                                  training_dataset: DataFrame,
                                                  classification:str,
                                                  frequency:str,
                                                  static_features:Optional[List[str]] = None )->DataFrame:
      
      step_name: str = self.__class__.__name__
       
      classification_artefact: ArtefactSpec = self._train_and_get_local_model_one_classification(training_dataset = training_dataset,
                                                                                                classification = classification,
                                                                                                frequency = frequency,
                                                                                                static_features = static_features)
      
      if classification_artefact is None:
         
        logger.info(f"[{step_name}] ArtefactSpec is None for classification='{classification}'. "
        "Returning empty Spark DataFrame.")

        empty_df: DataFrame = (
            training_dataset
            .select(F.lit(None).cast("string").alias("feature_name"),
                    F.lit(None).cast("double").alias("importance"),
                    F.lit(None).cast("string").alias("classification"))
            .limit(0)
        )

        return empty_df
         
         
      
      match classification_artefact.model_name:
         
          
        case 'XGBoostRegressor':
              
          feature_importance: List[float] = classification_artefact.local_model.feature_importances_

          if len(classification_artefact.features_columns) != len(feature_importance):
              
              raise ValueError(f"the quantity of feature importance: '{feature_importance}' has to be the same quiantity of features names: '{classification_artefact.features_columns}'. ")
              
          rows: List[Dict[str,float]] = list(zip(classification_artefact.features_columns, feature_importance))

          feature_importance_dataframe: DataFrame = self.spark.createDataFrame(rows, ['feature_name', 'importance'])

          return (feature_importance_dataframe
                  .withColumn('feature_name', F.col('feature_name').cast('string'))
                  .withColumn('importance', F.col('importance').cast('double'))
                  .withColumn('classification', F.lit(classification))
                )
              

        case 'LGBMRegressor':
              
          gain_importance: List[float] = classification_artefact.local_model.feature_importance('gain')

          total_gains:float = sum(gain_importance)

          feature_importance: List[float] = [gain / total_gains for gain in gain_importance]

          if len(classification_artefact.features_columns) != len(feature_importance):
              
              raise ValueError(f"the quantity of feature importance: '{feature_importance}' has to be the same quiantity of features names: '{classification_artefact.features_columns}'. ")
              
          rows: List[Dict[str,float]] = list(zip(classification_artefact.features_columns, feature_importance))

          feature_importance_dataframe: DataFrame = self.spark.createDataFrame(rows, ['feature_name', 'importance'])

          return (feature_importance_dataframe
                  .withColumn('feature_name', F.col('feature_name').cast('string'))
                  .withColumn('importance', F.col('importance').cast('double'))
                  .withColumn('classification', F.lit(classification))
                )
              
        case _:

          raise ValueError(f"Unsupported model name: {classification_artefact.model_name}")

        
         
    def feature_importance(self, 
                          training_dataset: DataFrame,
                          frequency: str,
                          static_features:Optional[List[str]] = None)->DataFrame:
       

      feature_importance_smooth: DataFrame = self._get_feature_importance_one_classification(training_dataset = training_dataset,
                                                                                             classification = 'Smooth',
                                                                                             frequency = frequency,
                                                                                             static_features = static_features)
      
      feature_importance_erratic: DataFrame = self._get_feature_importance_one_classification(training_dataset = training_dataset,
                                                                                              classification = 'Erratic',
                                                                                              frequency = frequency,
                                                                                              static_features = static_features)
      
      feature_importance_lumpy: DataFrame = self._get_feature_importance_one_classification(training_dataset = training_dataset,
                                                                                            classification = 'Lumpy',
                                                                                            frequency = frequency,
                                                                                            static_features = static_features)
      
      
      feature_importance_intermittent = self._get_feature_importance_one_classification(training_dataset = training_dataset,
                                                                                        classification = 'Intermittent',
                                                                                        frequency = frequency,
                                                                                        static_features = static_features)
      
      feature_importance_dataframe: DataFrame = (feature_importance_smooth
                                                .unionByName(feature_importance_erratic)
                                                .unionByName(feature_importance_lumpy)
                                                .unionByName(feature_importance_intermittent) 
                                                )
      
      return feature_importance_dataframe
       
        
           


   
       
       

      
      
      

        
      
      
        

        



    
        