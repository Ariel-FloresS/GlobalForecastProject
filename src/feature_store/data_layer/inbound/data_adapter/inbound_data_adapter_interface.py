from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class InboundDataAdapterInterface(ABC):

    """
    Interface for adapting inbound Spark DataFrames to the expected contract.

    The adapter validates required columns and standardizes types so downstream
    feature generation can assume a consistent schema.
    """

    @abstractmethod
    def inbound_adapter(self, input_dataframe:DataFrame)->DataFrame:
        raise NotImplementedError