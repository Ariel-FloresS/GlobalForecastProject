from .distributed_forecasting_engine_interface import DistributedForecastingEngineInterface
from pyspark.sql import DataFrame, SparkSession
from mlforecast.distributed.forecast import DistributedMLForecast
from model.business_layer.forecasting.model_factory import DistributedModel
from typing import List, Optional, Dict, Any, Self
import pandas as pd
import pyspark.sql.functions as F
from mlforecast.forecast import MLForecast


class DistributedForecastingEngine(DistributedForecastingEngineInterface):
   
   

    def __init__(self, models: List[DistributedModel], frequency: str,
                lags: Optional[List[int]] = None, lag_transforms: Optional[Dict[int, List[Any]]] = None,
                target_transforms: Optional[List[Any]] = None, num_threads: int = -1,
                max_future_rows_for_pandas: int = 2_000_000) -> None:
        
        self.distributed_ml_forecast = DistributedMLForecast( models = models,
                                                         freq = frequency,
                                                         lags = lags,
                                                         lag_transforms = lag_transforms,
                                                         target_transforms = target_transforms,
                                                         num_threads = num_threads)
        
        self._is_trained: bool = False
        self.max_future_rows_for_pandas = max_future_rows_for_pandas

    
    
    @classmethod
    def load(cls, path:str, engine:SparkSession)->Self:

        obj: Self = cls.__new__(cls)

        obj.distributed_ml_forecast = DistributedMLForecast.load(path = path, engine = engine)

        obj._is_trained = True
        
        obj.max_future_rows_for_pandas = 2_000_000

        return obj
    
    def _to_pandas_future_dataframe(self, future_dataframe: DataFrame)->pd.DataFrame:

        n_rows: int = future_dataframe.count()

        if n_rows > self.max_future_rows_for_pandas:

            raise ValueError(
                f"future_dataframe has {n_rows:,} rows; converting to pandas would be unsafe. "
                f"Increase max_future_rows_for_pandas or reduce scope (e.g., predict per class/batch)."
            )

        future_dataframe_pandas: pd.DataFrame = future_dataframe.toPandas()

        future_dataframe_pandas['ds'] = pd.to_datetime(future_dataframe_pandas['ds'])

        return future_dataframe_pandas


    def fit(self, training_dataset:DataFrame, static_features:Optional[List[str]] = None)->Self:

        training_dataset: DataFrame = training_dataset.withColumn("ds", F.to_timestamp("ds"))

        self.distributed_ml_forecast.fit(df =  training_dataset, static_features = static_features)

        self._is_trained = True

        return self
    
    def predict(self, prediction_horizon:int, future_dataframe:DataFrame)->DataFrame:

        if not self._is_trained:

            raise ValueError("DistributedMLForecast must be fitted or loaded before calling predict().")
        
        if prediction_horizon<=0:

            raise ValueError("prediction_horizon must be > 0.")
        
        future_dataframe_pandas: pd.DataFrame = self._to_pandas_future_dataframe(future_dataframe = future_dataframe)


        # X_df must be pandas per Nixtla's API contract
        return self.distributed_ml_forecast.predict(h = prediction_horizon, X_df = future_dataframe_pandas)
    
    def cross_validation(self, training_dataset:DataFrame, windows:int,
                        periods_for_each_window:int, static_features:Optional[List[str]] = None)->DataFrame:

        training_dataset: DataFrame = training_dataset.withColumn("ds", F.to_timestamp("ds"))
        
        cross_validation_dataframe: DataFrame = self.distributed_ml_forecast.cross_validation(df = training_dataset,
                                                                                              n_windows = windows,
                                                                                              h = periods_for_each_window,
                                                                                              static_features = static_features)
        return cross_validation_dataframe
    
    def save(self, path:str)->None:

        if not self._is_trained:

            raise ValueError("You can't save an unfitted model. Call fit() first.")
        
        self.distributed_ml_forecast.save(path = path)
        

    def to_local(self)->MLForecast:
        return self.distributed_ml_forecast.to_local()
    
    def preprocess(self, training_dataset: DataFrame, static_features:Optional[List[str]] = None)->DataFrame:


        training_dataset: DataFrame = training_dataset.withColumn("ds", F.to_timestamp("ds"))

        return self.distributed_ml_forecast.preprocess(df = training_dataset,
                                                       static_features = static_features)

        
        




      
        
    
    
   
      
    