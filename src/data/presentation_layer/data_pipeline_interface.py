from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class DataPipelineInterface(ABC):

    @abstractmethod
    def run_pipeline(self)->DataFrame:
        raise NotImplementedError

