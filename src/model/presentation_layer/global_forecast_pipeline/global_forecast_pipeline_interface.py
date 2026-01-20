from abc import ABC, abstractmethod
from typing import Optional, List
from pyspark.sql import DataFrame

class GlobalForecastPipelineInterface(ABC):

    @abstractmethod
    def forecast(self ,frequency:str, season_lenght:int, horizon:int, version:str ,static_features:Optional[List[str]] = None)->None:
        raise NotImplementedError
    
    @abstractmethod
    def backtesting(
                    self,
                    training_dataset:DataFrame,
                    frequency:str,
                    windows:int,
                    periods_for_each_window:int,
                    static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
