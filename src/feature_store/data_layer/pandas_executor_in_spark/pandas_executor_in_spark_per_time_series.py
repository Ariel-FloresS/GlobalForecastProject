from .pandas_executor_in_spark_per_time_series_interface import PandasExecutorInSparkPerTimeSeriesInterface
from pyspark.sql import DataFrame
import pandas as pd
from typing import Callable


class PandasExecutorInSparkPerTimeSeries(PandasExecutorInSparkPerTimeSeriesInterface):


    def apply_per_series(self, spark_dataframe_to_apply_the_pandas_function:DataFrame,
                        function: Callable[[pd.DataFrame], pd.DataFrame],
                        output_schema:str)->DataFrame:
        

        return (
            spark_dataframe_to_apply_the_pandas_function
            .groupBy('unique_id')
            .applyInPandas(func = function,
                           schema = output_schema)
        )

