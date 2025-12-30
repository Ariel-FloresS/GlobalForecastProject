from .training_data_repository_interface import TrainingDataRepositoryInterface
from pyspark.sql import DataFrame, SparkSession
from typing import List


class TrainingDataRepository(TrainingDataRepositoryInterface):

    def __init__(self, spark: SparkSession)->None:

        self.spark = spark
        self._required_columns = ['unique_id','ds','y','classification']


    def load_training_data(self, delta:str, exogenous_columns:List[str])->DataFrame:

        if not delta or not delta.strip():

            raise ValueError("delta must be a non-empty string.")

        loaded_data: DataFrame = self.spark.sql(f'SELECT * FROM {delta}')

        missing_required_columns: List[str] = [req for req in self._required_columns if req not in loaded_data.columns]
        
        if missing_required_columns:

            raise ValueError(f"Missing required columns '{missing_required_columns}' in the delta '{delta}'.")

        missing_exogenous_columns: List[str] = [missing for missing in exogenous_columns if missing not in loaded_data.columns]

        if missing_exogenous_columns:

            raise ValueError(f"Missing exogenous columns '{missing_exogenous_columns}' in delta table '{delta}'.")
        
        output_data: DataFrame = loaded_data.select(self._required_columns + exogenous_columns)

        return output_data
        

         


        
        