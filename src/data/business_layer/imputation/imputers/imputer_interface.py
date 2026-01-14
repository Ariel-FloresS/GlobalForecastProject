from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class ImputerInterface(ABC):

    @abstractmethod
    def impute(self, dataset:DataFrame)->DataFrame:
        raise NotImplementedError