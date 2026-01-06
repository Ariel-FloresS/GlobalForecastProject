from abc import ABC, abstractmethod
from pyspark.sql import SparkSession


class SparkSessionProviderInterface(ABC):

    @abstractmethod
    def get_spark_session(self)->SparkSession:
        raise NotImplementedError