from .exogenous_variable_interface import ExogenousVariableInterface
from feature_store.data_layer.pandas_executor_in_spark import PandasExecutorInSparkPerTimeSeriesInterface
from statsforecast.models import MSTL
from statsforecast.feature_engineering import mstl_decomposition
from pyspark.sql import DataFrame
import pandas as pd

class MSTLDecomposition(ExogenousVariableInterface):

    def __init__(self,
                executor: PandasExecutorInSparkPerTimeSeriesInterface,
                frequency:str,
                season_length:int, 
                horizon:int=0)->None:
        
        self.executor = executor
        self.frequency = frequency
        self.season_length = season_length
        self.horizon = horizon
        self._output_schema = """
                            unique_id string,
                            ds date,
                            trend double,
                            seasonal double
                            """
        
    def _mstl_decomposition_train(self,historical_pandas: pd.DataFrame)->pd.DataFrame:

        historical_pandas:pd.DataFrame = historical_pandas.sort_values('ds')

        train_dataframe, _ = mstl_decomposition(
                df = historical_pandas,
                model =  MSTL(season_length = self.season_length),
                freq = self.frequency,
                h = self.horizon + 1 
        )

        return train_dataframe.drop('y', axis = 1)
    
    def _mstl_decomposition_future(self,historical_pandas: pd.DataFrame)->pd.DataFrame:

        historical_pandas:pd.DataFrame = historical_pandas.sort_values('ds')

        _, X_df = mstl_decomposition(
                df = historical_pandas,
                model =  MSTL(season_length = self.season_length),
                freq = self.frequency,
                h = self.horizon 
        )

        return X_df
    

    def compute_exogenous(self, historical: DataFrame)->DataFrame:

        dataframe_base: DataFrame = historical.select('unique_id', 'ds', 'y')

        if self.horizon>0:

            mtsl_feats: DataFrame = self.executor.apply_per_series(
                spark_dataframe_to_apply_the_pandas_function = dataframe_base,
                function = self._mstl_decomposition_future,
                output_schema = self._output_schema
            )

        else:

            mtsl_feats: DataFrame = self.executor.apply_per_series(
                spark_dataframe_to_apply_the_pandas_function = dataframe_base,
                function = self._mstl_decomposition_train,
                output_schema = self._output_schema
            )

        return historical.join(other = mtsl_feats, on = ['unique_id', 'ds'], how='left')
