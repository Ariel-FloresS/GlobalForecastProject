from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional

class SegmentedForecastOrchestatorInterface(ABC):


    @abstractmethod
    def forecast(self, training_dataset: DataFrame, future_dataset: DataFrame, frequency:str,horizon: int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    