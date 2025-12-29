from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class DatasetPartitioningInterface(ABC):

    @abstractmethod
    def get_dataset_partition(self, dataset: DataFrame, partition_column:str)->DataFrame:
        raise NotImplementedError