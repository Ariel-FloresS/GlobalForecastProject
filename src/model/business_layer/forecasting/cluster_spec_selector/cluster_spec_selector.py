from .cluster_spec_selector_interface import ClusterSpecSelectorInterface
from forecasting.config import ModelSpec, MODEL_SPECS_BY_CLASS
from typing import Dict

class ClusterSpecSelector(ClusterSpecSelectorInterface):

    def __init__(self, model_specs_by_class:Dict[str, ModelSpec] = MODEL_SPECS_BY_CLASS)->None:

        self._model_specs_by_class = model_specs_by_class

    def get_spec_by_classification(self, classification:str)->ModelSpec:
        
        if classification not in self._model_specs_by_class.keys():
            raise ValueError(f"Unknow classification = '{classification}'. Allowed: {self._model_specs_by_class.keys()}")
        
        return self._model_specs_by_class[classification] 



