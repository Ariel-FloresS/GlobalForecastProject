from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional, Tuple

class RawDataRepositoryInterface(ABC):

    @abstractmethod
    def load_raw_data(self, delta:str, id_column:str, time_column:str, target_column:str, static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def train_test_split(self, split_date:str, delta:str, id_column:str, time_column:str, target_column:str, static_features:Optional[List[str]] = None)->Tuple[DataFrame, DataFrame]:
        raise NotImplementedError
    

