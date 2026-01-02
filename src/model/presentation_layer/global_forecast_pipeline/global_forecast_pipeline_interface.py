from abc import ABC, abstractmethod
from typing import Optional, List

class GlobalForecastPipelineInterface(ABC):

    @abstractmethod
    def forecast(self ,frequency:str, season_lenght:int, horizon:int, static_features:Optional[List[str]] = None)->None:
        raise NotImplementedError
