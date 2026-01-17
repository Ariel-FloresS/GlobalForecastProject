from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from typing import Optional, List


class GenerateFutureDatasetInterface(ABC):

    @abstractmethod
    def generate_dataset(self, spark: SparkSession, historical_dataframe:DataFrame,
                        horizon:int, frequency:str , static_features:Optional[List[str]] = None)->DataFrame:
        
        raise NotImplementedError

