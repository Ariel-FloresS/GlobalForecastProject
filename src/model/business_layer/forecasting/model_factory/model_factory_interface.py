from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List
from .distributed_model import DistributedModel

class ModelFactoryInterface(ABC):

    @abstractmethod
    def built_models(self, models_config: Dict[str, Dict[str,Any]])->List[DistributedModel]:
        raise NotImplementedError