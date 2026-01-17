from .training_data_repository_interface import TrainingDataRepositoryInterface
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional
from loguru import logger

class TrainingDataRepository(TrainingDataRepositoryInterface):

    def __init__(self, spark: SparkSession)->None:

        self.spark = spark
        self._required_columns = ['unique_id','ds','y','classification']


    def load_training_data(self, delta:str, exogenous_columns:List[str], static_features:Optional[List[str]] = None)->DataFrame:

        step_name: str = self.__class__.__name__

        if not delta or not delta.strip():

            raise ValueError("delta must be a non-empty string.")

        loaded_data: DataFrame = self.spark.sql(f'SELECT * FROM {delta}')

        static_features:List[str] = static_features or []

        necessary_columns: List[str] = self._required_columns + exogenous_columns + static_features

        missing_required_columns: List[str] = [req for req in necessary_columns if req not in loaded_data.columns]
        
        if missing_required_columns:

            raise ValueError(f"Missing required columns '{missing_required_columns}' in the delta '{delta}'.")

        
        output_data: DataFrame = loaded_data.select(necessary_columns)

        logger.info(f"[{step_name}] Training data loaded successfully.")

        return output_data
        

         


        
        