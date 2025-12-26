from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
import pandas as pd
from typing import Callable

class PandasExecutorInSparkPerTimeSeriesInterface(ABC):

    @abstractmethod
    def apply_per_series(self,spark_dataframe_to_apply_the_pandas_function:DataFrame,
                        function: Callable[[pd.DataFrame], pd.DataFrame],
                        output_schema:str)->DataFrame:
        raise NotImplementedError
    

