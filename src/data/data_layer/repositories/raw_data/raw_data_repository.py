from .raw_data_repository_interface import RawDataRepositoryInterface
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from loguru import logger
from typing import List

class RawDataRepository(RawDataRepositoryInterface):

    def __init__(self, spark:SparkSession)->None:
        self.spark = spark

    def load_raw_data(self, delta:str, id_column:str, time_column:str, target_column:str)->DataFrame:

        step_name: str = self.__class__.__name__

        if not delta or not delta.strip():

            raise ValueError(f'[{step_name}] delta must be a non-empty string.')

        input_dataframe: DataFrame = self.spark.sql(f'SELECT * FROM {delta}')

        required_cols: List[str] = [id_column, time_column, target_column]

        missing_cols: List[str] = [c for c in required_cols if c not in input_dataframe.columns]

        if missing_cols:

            raise ValueError(f"[{step_name}] Missing required columns '{missing_cols}' in delta '{delta}'.")
        

        input_dataframe: DataFrame = input_dataframe.select(required_cols)

        # Standardize to Nixtla schema
        out: DataFrame = (
            input_dataframe
            .select(
                F.col(id_column).cast('string').alias('unique_id'),
                F.to_date(F.col(time_column)).alias('ds'),
                F.col(target_column).cast('double').alias('y'),
            )
        )

        logger.info(f"[{step_name}] raw data loaded successfully.")
        return out
        
        