from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession



class GenerateFutureDatasetInterface(ABC):

    @abstractmethod
    def generate_dataset(self, spark:SparkSession ,historical_dataframe:DataFrame,
                        horizon:int, frequency:str)->DataFrame:
        
        raise NotImplementedError

