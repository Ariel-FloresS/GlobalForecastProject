from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import Optional, List

class FeatureStoreInterface(ABC):

    @abstractmethod
    def train_dataset(historical:DataFrame)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def future_dataset(self,historical: DataFrame, horizon:int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError