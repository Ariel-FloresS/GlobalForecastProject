from abc import ABC, abstractmethod
from pyspark.sql import DataFrame


class SegmentedImputationPipelineInterface(ABC):

    @abstractmethod
    def imputation(self, input_dataset: DataFrame)->DataFrame:
        raise NotImplementedError
