from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional
from model.data_layer.dtos import ArtefactSpec


class SegmentedForecastOrchestatorInterface(ABC):


    @abstractmethod
    def forecast(self, training_dataset: DataFrame, future_dataset: DataFrame, frequency:str,horizon: int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def cross_validation(self, training_dataset:DataFrame,frequency:str, windows:int,
                        periods_for_each_window:int, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    

    
    @abstractmethod
    def train_and_get_local_model(self,
                    training_dataset: DataFrame,
                    frequency:str,
                    static_features:Optional[List[str]] = None)->ArtefactSpec:
        raise NotImplementedError
    
    
    
    
    
    