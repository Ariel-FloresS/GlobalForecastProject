from .data_pipeline_interface import DataPipelineInterface
from data.data_layer.repositories.raw_data import RawDataRepositoryInterface, RawDataRepository
from data.data_layer.repositories.training_data import TrainingDataRepositoryInterface, TrainingDataRepository
from data.business_layer.data_cleaning_steps.steps import (
    FillMissingDatesStep, DropZeroOnlySeriesStep,RemoveLeadingNullsStep, 
    RemoveLeadingZeroesStep, DropInactiveRecentSeriesStep,DropShortSeriesStep,
    DataCleaningStepInterface
)
from data.business_layer.data_cleaning_steps.data_cleaning_pipeline import DataCleaningPipeline
from data.business_layer.classification import DemanClassifierFrepple
from data.business_layer.imputation.segmented_imputation_pipeline import SegmentedImputationPipeline
from data.business_layer.data_preparation_service import DataPreparationSeviceInterface, DataPreparationService
from feature_store.presentation_layer import FeatureStoreInterface, FeatureStore
from pyspark.sql import DataFrame, SparkSession, Row
import pyspark.sql.functions as F
from typing import List, Optional, Tuple
from loguru import logger



class DataPipeline(DataPipelineInterface):

    def __init__(self, spark: SparkSession, raw_delta_table: str, id_column:str,
                time_column:str, target_column:str, frequency:str, season_length:int,
                inactivity_periods:int, min_records_in_time_series: int, exogenous_columns:List[str],
                training_delta_table:str,
                static_features:Optional[List[str]] = None)->None:

        self.spark = spark
        self.raw_delta_table = raw_delta_table
        self.id_column = id_column
        self.time_column = time_column
        self.target_column = target_column
        self.frequency = frequency
        self.season_length = season_length
        self.inactivity_periods = inactivity_periods
        self.min_records_in_time_series = min_records_in_time_series
        self.exogenous_columns = exogenous_columns
        self.training_delta_table = training_delta_table
        self.static_features = static_features


    def _generate_cleaning_dataset(self, raw_dataset:DataFrame)->DataFrame:

        data_cleaning_steps: List[DataCleaningStepInterface] = [
            FillMissingDatesStep(spark = self.spark, frequency = self.frequency, static_features = self.static_features),
            RemoveLeadingNullsStep(),
            RemoveLeadingZeroesStep(),
            DropZeroOnlySeriesStep(),
            DropInactiveRecentSeriesStep(inactivity_periods = self.inactivity_periods),
            DropShortSeriesStep( min_records = self.min_records_in_time_series)
        ]

        data_preparation_Service: DataPreparationSeviceInterface = DataPreparationService(
            data_cleaning_pipeline = DataCleaningPipeline(cleaning_steps_list = data_cleaning_steps),
            classification_step = DemanClassifierFrepple(),
            segmented_imputation_pipeline = SegmentedImputationPipeline()
        )

        cleaning_dataset: DataFrame = data_preparation_Service.data_prepare(raw_dataset = raw_dataset)

        return cleaning_dataset


    def run_pipeline(self)->DataFrame:

        pipeline_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[DATA PIPELINE START]   {pipeline_name}\n{'='*84}"
        logger.info(banner_top)

        raw_data_repository: RawDataRepositoryInterface = RawDataRepository(spark = self.spark)

        raw_dataset: DataFrame = raw_data_repository.load_raw_data(delta = self.raw_delta_table,
                                                                id_column = self.id_column,
                                                                time_column = self.time_column,
                                                                target_column = self.target_column,
                                                                static_features = self.static_features)
        

        
        cleaning_dataset: DataFrame = self._generate_cleaning_dataset(raw_dataset = raw_dataset)
        

        feature_store: FeatureStoreInterface = FeatureStore(spark = self.spark, 
                                                            frequency = self.frequency,
                                                            season_length = self.season_length
                                                            )

        training_dataset: DataFrame = feature_store.train_dataset(historical = cleaning_dataset)

        TrainingDataRepository().save_training_data(training_dataframe = training_dataset,
                                                    delta = self.training_delta_table,
                                                    exogenous_columns = self.exogenous_columns)
        

        banner_end: str = f"\n{'='*84}\n[DATA PIPELINE END]   {pipeline_name}\n{'='*84}"
        logger.info(banner_end)
        
    def train_test_future_datasets_split(self, split_date:str, horizon:int)->Tuple[DataFrame, DataFrame, DataFrame]:

        pipeline_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[TRAIN, TEST, FUTURE SPLIT START]   {pipeline_name}\n{'='*84}"
        logger.info(banner_top)

        raw_data_repository: RawDataRepositoryInterface = RawDataRepository(spark = self.spark)

        train_dataset_raw, test_dataset_raw = raw_data_repository.train_test_split(split_date = split_date,
                                                                                    delta = self.raw_delta_table,
                                                                                    id_column = self.id_column,
                                                                                    time_column = self.time_column,
                                                                                    target_column = self.target_column,
                                                                                    static_features = self.static_features )
        
        train_cleaning_dataset: DataFrame = self._generate_cleaning_dataset(raw_dataset = train_dataset_raw)

        feature_store: FeatureStoreInterface = FeatureStore(spark = self.spark, 
                                                            frequency = self.frequency,
                                                            season_length = self.season_length
                                                            )
        
        training_dataset: DataFrame = feature_store.train_dataset(historical = train_cleaning_dataset)

        future_dataset: DataFrame = feature_store.future_dataset(historical = train_cleaning_dataset,
                                                                horizon = horizon,
                                                                static_features = self.static_features)


        bounds_train: Row = (
            training_dataset
            .select(F.min("ds").alias("min_ds"), F.max("ds").alias("max_ds"))
            .first()
        )
        bounds_test: Row = (
            test_dataset_raw
            .select(F.min("ds").alias("min_ds"), F.max("ds").alias("max_ds"))
            .first()
        )

        bounds_future: Row = (
            future_dataset
            .select(F.min("ds").alias("min_ds"), F.max("ds").alias("max_ds"))
            .first()
        )


        logger.info(
            f'[{pipeline_name}] Dataset ranges:\n'
            f'  • TRAIN  | min_ds={bounds_train["min_ds"]} | max_ds={bounds_train["max_ds"]}\n'
            f'  • TEST   | min_ds={bounds_test["min_ds"]} | max_ds={bounds_test["max_ds"]}\n'
            f'  • FUTURE | min_ds={bounds_future["min_ds"]} | max_ds={bounds_future["max_ds"]}'
        )

        banner_end: str = (
            f'\n{"="*84}\n'
            f'[PIPELINE END] TRAIN / TEST / FUTURE SPLIT | {pipeline_name}\n'
            f'{"="*84}'
        )

        logger.info(banner_end)


        return training_dataset, test_dataset_raw, future_dataset



        



        
        
    

        




        




    