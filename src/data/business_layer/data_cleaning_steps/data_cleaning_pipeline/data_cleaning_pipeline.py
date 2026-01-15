from .data_cleaning_pipeline_interface import DataCleaningPipelineInterface
from data.business_layer.data_cleaning_steps.steps import DataCleaningStepInterface
from pyspark.sql import DataFrame
from loguru import logger
from typing import List


class DataCleaningPipeline(DataCleaningPipelineInterface):

    def __init__(self, cleaning_steps_list: List[DataCleaningStepInterface])->None:

        self.cleaning_steps_list = cleaning_steps_list
        

    def cleaning(self, dataset:DataFrame)->DataFrame:

        pipeline_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[PIPELINE START]   {pipeline_name}\n{'='*84}"
        logger.info(banner_top)

        if not self.cleaning_steps_list:
            logger.warning(f"[PIPELINE] {pipeline_name} | No steps provided. Returning dataset unchanged.")
            return dataset
     
        
        for step in self.cleaning_steps_list:

            if not isinstance(step, DataCleaningStepInterface):
                raise TypeError(
                    f"[PIPELINE] {pipeline_name}: step ({type(step).__name__}) must implement DataCleaningStepInterface."
                )
            
            dataset: DataFrame = step.apply_transformation(input_dataframe = dataset)

        banner_bottom: str = f"\n{'='*84}\n[PIPELINE END]   {pipeline_name}\n{'='*84}"
        logger.info(banner_bottom)

        return dataset


