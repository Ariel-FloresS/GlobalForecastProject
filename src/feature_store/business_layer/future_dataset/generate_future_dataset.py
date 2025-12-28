from .generate_future_dataset_interface import GenerateFutureDatasetInterface
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pandas as pd
import datetime 
from typing import List


class GenerateFutureDataset(GenerateFutureDatasetInterface):

    def generate_dataset(self, spark: SparkSession, historical_dataframe:DataFrame,
                        horizon:int, frequency:str)->DataFrame:
        

        ids_dataframe: DataFrame = historical_dataframe.select('unique_id').distinct()

        start_date: datetime.date = historical_dataframe.select(F.max("ds").alias("max_ds")).first()["max_ds"]

        date_range: pd.DatetimeIndex = pd.date_range(start = start_date,
                                                    periods = horizon+1,
                                                    freq = frequency)
        
        date_range: pd.DatetimeIndex = date_range.drop( date_range.min() )
        
        date_list: List[str] = [(str(date),) for date in date_range]

        dates_dataframe: DataFrame = spark.createDataFrame( date_list, ['ds'] )

        future_dataset: DataFrame = ids_dataframe.crossJoin(other = dates_dataframe)

        return future_dataset