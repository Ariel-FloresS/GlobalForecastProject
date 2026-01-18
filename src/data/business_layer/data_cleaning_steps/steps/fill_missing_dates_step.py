from .data_cleaning_step_interface import DataCleaningStepInterface
from pyspark.sql import DataFrame, SparkSession,Row
import pyspark.sql.functions as F
import pandas as pd
import datetime
from loguru import logger
from typing import List, Optional

class FillMissingDatesStep(DataCleaningStepInterface):

    def __init__(self, spark:SparkSession, frequency:str, static_features:Optional[List[str]] = None)->None:

        self.spark = spark
        self.frequency = frequency
        self.static_features = static_features
        

    def apply_transformation(self, input_dataframe:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name}")

        list_static_features:List[str] = self.static_features or []

        static_columns: List[str] = ['unique_id'] + list_static_features
        
        ids_dataframe: DataFrame = input_dataframe.select(static_columns).distinct()

        bounds: Row = (
            input_dataframe
            .select(F.min('ds').alias('min_ds'), F.max('ds').alias('max_ds'))
            .first()
        )

        start_date: datetime.date = bounds['min_ds']
        end_date: datetime.date = bounds['max_ds']

        date_range: pd.DatetimeIndex = pd.date_range(start=start_date, end=end_date, freq=self.frequency)

        
        pdf_dates:pd.DataFrame = pd.DataFrame({'ds': pd.to_datetime(date_range).date})  
        dates_dataframe: DataFrame = self.spark.createDataFrame(pdf_dates)  

        
        dates_dataframe = F.broadcast(dates_dataframe)

        complete_dates_dataframe: DataFrame = ids_dataframe.crossJoin(dates_dataframe)

        right_dataframe: DataFrame = input_dataframe.drop(*list_static_features)

        output_dataframe: DataFrame = complete_dates_dataframe.join(
            other = right_dataframe, on = ['unique_id', 'ds'], how = 'leftouter'
        )
        
        logger.info(f"Finishing: {step_name}")

        return output_dataframe
        