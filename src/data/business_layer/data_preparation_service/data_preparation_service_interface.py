from abc import ABC, abstractmethod
from pyspark.sql import DataFrame


class DataPreparationSeviceInterface(ABC):

    @abstractmethod
    def data_prepare(self, raw_dataset: DataFrame)->DataFrame:
        raise NotImplementedError