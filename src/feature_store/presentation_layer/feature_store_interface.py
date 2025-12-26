from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class FeatureStoreInterface(ABC):

    @abstractmethod
    def train_dataset(historical:DataFrame)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def future_dataset(historical:DataFrame, horizon:int)->DataFrame:
        raise NotImplementedError