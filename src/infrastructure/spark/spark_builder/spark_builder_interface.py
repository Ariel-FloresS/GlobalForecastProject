from abc import ABC, abstractmethod
from pyspark.sql import SparkSession

class SparkBuilderInterface(ABC):

    
    @abstractmethod
    def build_spark(self)->SparkSession:
        raise NotImplementedError