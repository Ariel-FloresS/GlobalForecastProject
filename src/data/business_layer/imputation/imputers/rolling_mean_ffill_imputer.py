from imputation.imputer_interface import ImputerInterface
from pyspark.sql import DataFrame, Window, Column
import pyspark.sql.functions as F
from loguru import logger


class RollingMeanFFillImputer(ImputerInterface):

    def __init__(self, window_size: int, fallback_value: float = 0.0)->None:

        self.window_size = window_size
        self.fallback_value = fallback_value

    def impute(self, dataset:DataFrame)->DataFrame:

        

        step_name: str = self.__class__.__name__

        logger.info(f"Starting:  {step_name}: imputing with rolling mean + ffill (k={self.window_size}).")

        if self.window_size < 1: raise ValueError('window_size must be >= 1')
        
        w_roll: Window = (
            Window.partitionBy('unique_id')
                  .orderBy('ds')
                  .rowsBetween(-(self.window_size - 1), 0)
        )

        w_ffill: Window = (
            Window.partitionBy('unique_id')
                  .orderBy('ds')
                  .rowsBetween(Window.unboundedPreceding, 0)
        )
        
        rolling_mean: Column =  F.avg(F.col('y').cast("double")).over(w_roll)

        ffill: Column = F.last(F.col('y'), ignorenulls=True).over(w_ffill)


        out: DataFrame = (
            dataset
            .withColumn(
                'y',
                F.coalesce(F.col('y'), rolling_mean, ffill, F.lit(self.fallback_value).cast("double"))
            )
        )

        logger.info(f"Finished: {step_name}")

        return out
        