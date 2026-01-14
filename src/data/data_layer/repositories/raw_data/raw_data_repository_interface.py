from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class RawDataRepositoryInterface(ABC):

    @abstractmethod
    def load_raw_data(self, delta:str, id_column:str, time_column:str, target_column:str)->DataFrame:
        raise NotImplementedError
