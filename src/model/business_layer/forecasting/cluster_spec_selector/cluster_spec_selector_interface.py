from abc import ABC, abstractmethod
from model.business_layer.forecasting.config import ModelSpec

class ClusterSpecSelectorInterface(ABC):

    @abstractmethod
    def get_spec_by_classification(self, classification:str)->ModelSpec:
        raise NotImplementedError
