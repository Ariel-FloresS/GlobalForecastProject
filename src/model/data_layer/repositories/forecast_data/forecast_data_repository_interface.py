from abc import ABC, abstractmethod
from pyspark.sql import DataFrame


class ForecastDataRepositoryInterface(ABC):
    
    @abstractmethod
    def save_forecast_data(self, forecast_dataframe:DataFrame, version: str ,delta:str)->None:
        raise NotImplementedError