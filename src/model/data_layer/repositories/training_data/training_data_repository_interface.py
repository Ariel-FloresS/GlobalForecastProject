from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Optional

class TrainingDataRepositoryInterface(ABC):

    @abstractmethod
    def load_training_data(self, delta:str, exogenous_columns:List[str], static_features:Optional[List[str]] = None)->DataFrame:
        raise NotImplementedError