from .exogenous_variable_interface import ExogenousVariableInterface
from feature_store.data_layer.pandas_executor_in_spark import PandasExecutorInSparkPerTimeSeriesInterface
from statsforecast.models import MSTL
from statsforecast.feature_engineering import mstl_decomposition
from pyspark.sql import DataFrame
import pandas as pd

class MSTLDecomposition(ExogenousVariableInterface):

    def __init__(self,
                executor: PandasExecutorInSparkPerTimeSeriesInterface,
                training_dataframe:DataFrame,
                frequency:str,
                season_length:int, 
                horizon:int=0)->None:
        
        self.executor = executor
        self.training_dataframe = training_dataframe
        self.frequency = frequency
        self.season_length = season_length
        self.horizon = horizon
        self._output_schema = """
                            unique_id string,
                            ds date,
                            trend double,
                            seasonal double
                            """
        


    def compute_exogenous(self, historical: DataFrame)->DataFrame:

        dataframe_base: DataFrame = self.training_dataframe.select('unique_id', 'ds', 'y')

        freq: str = self.frequency
        season_length: int = self.season_length
        horizon: int = self.horizon

        if self.horizon>0:

            def future_wrapper(unique_id:str, pdf:pd.DataFrame)->pd.DataFrame:

                pdf: pd.DataFrame = pdf.sort_values('ds')

                _, X_df = mstl_decomposition(
                    df = pdf,
                    model =  MSTL(season_length = season_length),
                    freq = freq,
                    h = horizon 
                )
                return X_df

                 

            mtsl_feats: DataFrame = self.executor.apply_per_series(
                spark_dataframe_to_apply_the_pandas_function = dataframe_base,
                function = future_wrapper,
                output_schema = self._output_schema
            )

        else:

            def train_wrapper(unique_id: str, pdf:pd.DataFrame)->pd.DataFrame:

                pdf: pd.DataFrame = pdf.sort_values('ds')

                train_dataframe, _ = mstl_decomposition(
                    df = pdf,
                    model =  MSTL(season_length = season_length),
                    freq = freq,
                    h = 1 
                )

                return train_dataframe.drop('y', axis = 1)


            mtsl_feats: DataFrame = self.executor.apply_per_series(
                spark_dataframe_to_apply_the_pandas_function = dataframe_base,
                function = train_wrapper,
                output_schema = self._output_schema
            )

        return historical.join(other = mtsl_feats, on = ['unique_id', 'ds'], how='left')
