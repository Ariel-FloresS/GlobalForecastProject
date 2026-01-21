from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List, Tuple
from .distributed_model import DistributedModel
from .local_model import LocalModel

class ModelFactoryInterface(ABC):

    @abstractmethod
    def built_models(self, models_config: Dict[str, Dict[str,Any]])->List[DistributedModel]:
        raise NotImplementedError
    
    @abstractmethod
    def built_local_model(self, distributed_model: Dict[DistributedModel])->Tuple[str,LocalModel]:
        raise NotImplementedError