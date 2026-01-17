from .feature_store_interface import FeatureStoreInterface
from feature_store.business_layer.feature_service import FeatureServiceInterface, FeatureService
from feature_store.data_layer.pandas_executor_in_spark import PandasExecutorInSparkPerTimeSeries
from feature_store.data_layer.inbound.data_adapter import InboundDataAdapter
from feature_store.business_layer.exogenous import MSTLDecomposition, ExogenousVariableInterface
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional


class FeatureStore(FeatureStoreInterface):

    def __init__(self, spark: SparkSession, frequency:str, season_length:int, static_features:Optional[List[str]] = None)->None:
        self.spark = spark
        self.frequency = frequency
        self.season_length = season_length
        self.pandas_executor = PandasExecutorInSparkPerTimeSeries()
        self.inbound_adapter = InboundDataAdapter()
        self.static_features = static_features

    def train_dataset(self, historical: DataFrame)->DataFrame:

        exogenous_variables: List[ExogenousVariableInterface] = [

            MSTLDecomposition(executor = self.pandas_executor,
                              training_dataframe = historical,
                              frequency = self.frequency,
                              season_length = self.season_length)
        ]

        feature_service: FeatureServiceInterface = FeatureService(spark = self.spark,
                                                                  list_exogenous_variables = exogenous_variables,
                                                                  inbound_adapter = self.inbound_adapter)
        
        return feature_service.generate_train_dataset(historical = historical)

    def future_dataset(self,historical: DataFrame, horizon:int)->DataFrame:

        exogenous_variables: List[ExogenousVariableInterface] = [

            MSTLDecomposition(executor = self.pandas_executor,
                              training_dataframe = historical,
                              frequency = self.frequency,
                              season_length = self.season_length,
                              horizon = horizon)
            ]
        
        feature_service: FeatureServiceInterface = FeatureService(spark = self.spark,
                                                                  list_exogenous_variables = exogenous_variables,
                                                                  inbound_adapter = self.inbound_adapter)
        
        return feature_service.generate_future_dataset(spark = self.spark,
                                                       historical = historical,
                                                       horizon = horizon,
                                                        frequency = self.frequency,
                                                        static_features = self.static_features )


        

        

    

        

