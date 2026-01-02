from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass(frozen = True)
class ModelSpec:
    name:str
    lags: Optional[List[int]] = None
    lag_transforms: Optional[Dict[int, List[Any]]] = None
    target_transforms: Optional[List[Any] ]= None
    models:Dict[str, Dict[str, Any]] = field(default_factory = dict)
