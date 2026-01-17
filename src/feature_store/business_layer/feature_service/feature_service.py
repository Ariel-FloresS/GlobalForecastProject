from .feature_service_interface import FeatureServiceInterface
from feature_store.data_layer.inbound.data_adapter import InboundDataAdapterInterface
from feature_store.business_layer.exogenous import ExogenousVariableInterface
from feature_store.business_layer.future_dataset import GenerateFutureDatasetInterface, GenerateFutureDataset
from pyspark.sql import DataFrame, SparkSession
from typing import List, Optional
from loguru import logger

class FeatureService(FeatureServiceInterface):

    def __init__(self, spark:SparkSession,
                list_exogenous_variables:List[ExogenousVariableInterface],
                inbound_adapter: InboundDataAdapterInterface)->None:
        
        self.spark = spark
        self.list_exogenous_variables = list_exogenous_variables
        self.inbound_adapter = inbound_adapter
        

    def generate_train_dataset(self, historical: DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"[{step_name}] Generating Exogenous Variables For Training Dataset")

        historical: DataFrame = self.inbound_adapter.inbound_adapter(input_dataframe = historical)

        for exogenous in self.list_exogenous_variables:

            historical = exogenous.compute_exogenous(historical = historical)

        return historical
    
    def generate_future_dataset(self, spark:SparkSession, historical: DataFrame,
                                horizon: int, frequency: str, static_features:Optional[List[str]] = None)->DataFrame:
        step_name: str = self.__class__.__name__
        
        logger.info(f"[{step_name}] Generating Exogenous Variables For Future Dataset")
        
        generate_future_dataset:GenerateFutureDatasetInterface = GenerateFutureDataset()

        historical: DataFrame = self.inbound_adapter.inbound_adapter(input_dataframe = historical)

        future_data: DataFrame = generate_future_dataset.generate_dataset(spark = spark,
                                                                          historical_dataframe = historical,
                                                                          horizon = horizon,
                                                                          frequency = frequency,
                                                                          static_features = static_features)
        
        for exogenous in self.list_exogenous_variables:

            future_data = exogenous.compute_exogenous(historical = future_data)

        return future_data


        
        
        

        

        

        
    
   
        
        
        