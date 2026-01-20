from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional

class SegmentedForecastOrchestatorInterface(ABC):


    @abstractmethod
    def forecast(self, training_dataset: DataFrame, future_dataset: DataFrame, frequency:str,horizon: int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def cross_validation(self, training_dataset:DataFrame,frequency:str, windows:int,
                        periods_for_each_window:int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    
    
    