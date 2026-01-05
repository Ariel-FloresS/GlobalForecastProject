from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional, Self

class DistributedForecastingEngineInterface(ABC):

    @abstractmethod
    def fit(self, training_dataset:DataFrame, static_features:Optional[List[str]] = None)->Self:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, prediction_horizon:int, future_dataframe:DataFrame)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def cross_validation(self,training_dataset:DataFrame, windows:int,
                        periods_for_each_window:int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path:str)->None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, engine: SparkSession) -> Self:
        raise NotImplementedError