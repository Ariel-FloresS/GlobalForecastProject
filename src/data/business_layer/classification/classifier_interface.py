from abc import ABC, abstractmethod
from pyspark.sql import DataFrame


class ClassifierInterface(ABC):
    

    @abstractmethod
    def classify(self, dataset:DataFrame)->DataFrame:
        raise NotImplementedError