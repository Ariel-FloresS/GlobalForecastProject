from .data_preparation_service_interface import DataPreparationSeviceInterface
from data.business_layer.data_cleaning_steps.data_cleaning_pipeline import DataCleaningPipelineInterface
from data.business_layer.classification import ClassifierInterface
from data.business_layer.imputation.segmented_imputation_pipeline import SegmentedImputationPipelineInterface
from pyspark.sql import DataFrame, SparkSession


class DataPreparationService(DataPreparationSeviceInterface):

    def __init__(self, data_cleaning_pipeline: DataCleaningPipelineInterface,
                classification_step: ClassifierInterface, segmented_imputation_pipeline: SegmentedImputationPipelineInterface)->None:

        self.data_cleaning_pipeline = data_cleaning_pipeline
        self.classification_step = classification_step
        self.segmented_imputation_pipeline = segmented_imputation_pipeline


    def data_prepare(self, raw_dataset: DataFrame)->DataFrame:

        cleaning_dataframe: DataFrame = self.data_cleaning_pipeline.cleaning(dataset = raw_dataset)

        classify_dataframe: DataFrame = self.classification_step.classify(dataset = cleaning_dataframe)

        imputation_dataframe: DataFrame = self.segmented_imputation_pipeline.imputation(input_dataset = classify_dataframe)

        return imputation_dataframe



