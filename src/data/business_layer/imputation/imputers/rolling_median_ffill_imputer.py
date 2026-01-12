from .imputer_interface import ImputerInterface
from pyspark.sql import DataFrame, Window, Column
import pyspark.sql.functions as F
from loguru import logger


class RollingMedianFFillImputer(ImputerInterface):

    def __init__(self, window_size: int = 6, fallback_value: float = 0.0)->None:

        self.window_size = window_size
        self.fallback_value = fallback_value

    def impute(self, dataset:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name}: imputing with rolling median + ffill (k={self.window_size}).")

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

        rolling_median: Column = F.percentile_approx(F.col('y').cast("double"), 0.5).over(w_roll)

        ffill: Column = F.last(F.col('y'), ignorenulls=True).over(w_ffill)


        out: DataFrame = (
            dataset
            .withColumn(
                'y',
                F.coalesce(F.col('y'), rolling_median, ffill, F.lit(self.fallback_value).cast("double"))
            )
        )

        logger.info(f"Finished: {step_name}")
        return out
        