from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen = True)
class ModelSpec:
    name:str
    models:Dict[str, Dict[str, Any]] = field(default = dict)