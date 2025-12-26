from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession




class FeatureServiceInterface(ABC):

    @abstractmethod
    def generate_train_dataset(self, historical:DataFrame)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def generate_future_dataset(self, spark:SparkSession, historical:DataFrame, horizon:int, frequency:str )->DataFrame:
        raise NotImplementedError