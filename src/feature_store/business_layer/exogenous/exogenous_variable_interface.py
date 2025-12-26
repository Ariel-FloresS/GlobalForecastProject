from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class ExogenousVariableInterface(ABC):

    @abstractmethod
    def compute_exogenous(self, historical:DataFrame)->DataFrame:
        raise NotImplementedError