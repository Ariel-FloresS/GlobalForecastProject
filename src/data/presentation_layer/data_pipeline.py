from .data_pipeline_interface import DataPipelineInterface
from data_layer.repositories.raw_data import RawDataRepositoryInterface, RawDataRepository
from data_layer.repositories.training_data import TrainingDataRepositoryInterface, TrainingDataRepository
from business_layer.data_cleaning_steps.steps import (
    FillMissingDatesStep, DropZeroOnlySeriesStep,RemoveLeadingNullsStep, 
    RemoveLeadingZeroesStep, DropInactiveRecentSeriesStep,DropShortSeriesStep,
    DataCleaningStepInterface
)
from business_layer.data_cleaning_steps.data_cleaning_pipeline import DataCleaningPipeline
from business_layer.classification import DemanClassifierFrepple
from business_layer.imputation.segmented_imputation_pipeline import SegmentedImputationPipeline
from business_layer.data_preparation_service import DataPreparationSeviceInterface, DataPreparationService
from feature_store.presentation_layer import FeatureStoreInterface, FeatureStore
from pyspark.sql import DataFrame, SparkSession
from typing import List


class DataPipeline(DataPipelineInterface):

    def __init__(self, spark: SparkSession, raw_delta_table: str, id_column:str,
                time_column:str, target_column:str, frequency:str, season_length:int,
                inactivity_periods:int, min_records_in_time_series: int, exogenous_columns:List[str],
                training_delta_table:str)->None:

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

    def run_pipeline(self)->DataFrame:

        raw_data_repository: RawDataRepositoryInterface = RawDataRepository(spark = self.spark)

        raw_dataset: DataFrame = raw_data_repository.load_raw_data(delta = self.raw_delta_table,
                                                                id_column = self.id_column,
                                                                time_column = self.time_column,
                                                                target_column = self.target_column)
        data_cleaning_steps: List[DataCleaningStepInterface] = [
            FillMissingDatesStep(spark = self.spark, frequency = self.frequency),
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

        feature_store: FeatureStoreInterface = FeatureStore(spark = self.spark, 
                                                            frequency = self.frequency,
                                                            season_length = self.season_length
                                                            )

        training_dataset: DataFrame = feature_store.train_dataset(historical = cleaning_dataset)

        TrainingDataRepository().save_training_data(training_dataframe = training_dataset,
                                                    delta = self.training_delta_table,
                                                    exogenous_columns = self.exogenous_columns)

        




        




    