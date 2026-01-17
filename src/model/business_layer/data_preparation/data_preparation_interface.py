from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import Tuple, List, Optional


class DataPreparationInterface(ABC):

    
    @abstractmethod
    def prepare_batch_training_and_future_datasets(self, training_delta_table:str,
                                                   exogenous_columns: List[str],
                                                   horizon: int,
                                                   static_features:Optional[List[str]] = None)->Tuple[DataFrame, DataFrame]:
        raise NotImplementedError