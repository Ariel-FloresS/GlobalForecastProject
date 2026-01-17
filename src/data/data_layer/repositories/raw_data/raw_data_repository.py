from .raw_data_repository_interface import RawDataRepositoryInterface
from pyspark.sql import DataFrame, SparkSession, Column
import pyspark.sql.functions as F
from loguru import logger
from typing import List, Optional, Tuple


class RawDataRepository(RawDataRepositoryInterface):

    def __init__(self, spark:SparkSession)->None:
        self.spark = spark

    def load_raw_data(self, delta:str, id_column:str, time_column:str, target_column:str, static_features:Optional[List[str]] = None)->DataFrame:

        step_name: str = self.__class__.__name__

        if not delta or not delta.strip():

            raise ValueError(f'[{step_name}] delta must be a non-empty string.')

        input_dataframe: DataFrame = self.spark.sql(f'SELECT * FROM {delta}')

        static_features:List[str] = static_features or []

        required_cols: List[str] = [id_column, time_column, target_column] + static_features

        missing_cols: List[str] = [c for c in required_cols if c not in input_dataframe.columns]

        if missing_cols:

            raise ValueError(f"[{step_name}] Missing required columns '{missing_cols}' in delta '{delta}'.")
        

        input_dataframe: DataFrame = input_dataframe.select(required_cols)

        # Standardize to Nixtla schema
        select_exprs: List[Column]  = [
            F.col(id_column).cast("string").alias("unique_id"),
            F.to_date(F.col(time_column)).alias("ds"),
            F.col(target_column).cast("double").alias("y"),
        ] + [F.col(c) for c in static_features]

        
        out: DataFrame = (
            input_dataframe
            .select(*select_exprs)
        )

        logger.info(f"[{step_name}] raw data loaded successfully.")
        return out
    
    def train_test_split(self, split_date:str, delta:str, id_column:str, time_column:str,
                        target_column:str, static_features:Optional[List[str]] = None)->Tuple[DataFrame, DataFrame]:
        
        step_name: str = self.__class__.__name__
        
        if not split_date or not str(split_date).strip():

            raise ValueError(f"[{step_name}] split_date must be a non-empty string (e.g. '2024-12-31').")
        
        input_dataframe: DataFrame = self.load_raw_data(delta = delta,
                                                        id_column = id_column,
                                                        time_column =  time_column,
                                                        target_column = target_column,
                                                        static_features = static_features)
        
        train_dataframe_raw: DataFrame = input_dataframe.filter(F.col('ds') <= split_date)

        test_dataframe_raw: DataFrame = input_dataframe.filter(F.col('ds') > split_date)

        train_has_rows: bool = train_dataframe_raw.limit(1).count() > 0
        test_has_rows: bool = test_dataframe_raw.limit(1).count() > 0

        if not train_has_rows or not test_has_rows:
            
            raise ValueError(
                f"[{step_name}] Split produced empty partition(s). "
                f"train_empty={not train_has_rows}, test_empty={not test_has_rows}, split_date={split_date}."
            )
        


        return train_dataframe_raw, test_dataframe_raw
