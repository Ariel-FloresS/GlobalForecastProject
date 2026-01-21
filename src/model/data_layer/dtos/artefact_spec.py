from dataclasses import dataclass, field
from typing import Any, List

@dataclass(frozen = True)
class ArtefactSpec:
    
    model_name:str
    classification:str
    local_model: Any
    features_columns: List[str] = field(default_factory = list)
    