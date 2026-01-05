from .data_preparation_interface import DataPreparationInterface
from model.data_layer.repositories.training_data import  TrainingDataRepositoryInterface
from model.data_layer.dataset_partitioning import DatasetPartitioningInterface
from feature_store.presentation_layer import FeatureStoreInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from typing import Tuple, List


class DataPreparation(DataPreparationInterface):

    def __init__(self,
                training_repository: TrainingDataRepositoryInterface,
                dataset_partitioning: DatasetPartitioningInterface,
                feature_store: FeatureStoreInterface,)->None:


        self.training_repository = training_repository
        self.dataset_partitioning = dataset_partitioning
        self.feature_store = feature_store
        self._partition_column = 'unique_id'
    

    def prepare_batch_training_and_future_datasets(self, training_delta_table: str,
                                                    exogenous_columns: List[str], horizon:int)->Tuple[DataFrame, DataFrame]:
        
        training_dataframe: DataFrame = self.training_repository.load_training_data(delta = training_delta_table,
                                                                                    exogenous_columns = exogenous_columns)
        
        training_dataset: DataFrame = self.dataset_partitioning.get_dataset_partition(dataset = training_dataframe,
                                                                                    partition_column = self._partition_column)
        
        future_dataset: DataFrame = self.feature_store.future_dataset(historical =  training_dataset, 
                                                                      horizon =  horizon)
        
        training_dataset: DataFrame = training_dataset.withColumn('ds', F.to_timestamp('ds'))
        future_dataset: DataFrame = future_dataset.withColumn('ds', F.to_timestamp('ds'))

        return training_dataset, future_dataset
        

        

        
        
        
        


        

        

    
        
