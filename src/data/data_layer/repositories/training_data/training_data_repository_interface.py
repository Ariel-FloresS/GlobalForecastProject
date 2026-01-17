from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional

class TrainingDataRepositoryInterface(ABC):

    @abstractmethod
    def save_training_data(self,training_dataframe:DataFrame, delta:str, exogenous_columns:List[str], static_features:Optional[List[str]] = None)->None:
        raise NotImplementedError