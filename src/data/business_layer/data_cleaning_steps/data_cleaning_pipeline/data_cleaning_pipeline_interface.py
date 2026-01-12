from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class DataCleaningPipelineInterface(ABC):

    @abstractmethod
    def cleaning(self, dataset:DataFrame)->DataFrame:
        raise NotImplementedError