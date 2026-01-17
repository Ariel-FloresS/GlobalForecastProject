from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional



class FeatureServiceInterface(ABC):

    @abstractmethod
    def generate_train_dataset(self, historical:DataFrame)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def generate_future_dataset(self, spark:SparkSession, historical:DataFrame,
                                horizon:int, frequency:str, static_features:Optional[List[str]] = None )->DataFrame:
        raise NotImplementedError