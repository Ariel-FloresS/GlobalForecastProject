from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class DataCleaningStepInterface(ABC):

    @abstractmethod
    def apply_transformation(self, input_dataframe:DataFrame)->DataFrame:
        raise NotImplementedError
    
