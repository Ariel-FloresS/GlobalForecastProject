from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import Tuple

class DataPipelineInterface(ABC):

    @abstractmethod
    def run_pipeline(self)->DataFrame:
        raise NotImplementedError
    
    def train_test_future_datasets_split(self, split_date:str, horizon:int)->Tuple[DataFrame, DataFrame, DataFrame]:
        raise NotImplementedError

