from .imputer_interface import ImputerInterface
from pyspark.sql import DataFrame, Window, Column
import pyspark.sql.functions as F
from loguru import logger

class ZeroFillImputer(ImputerInterface):

    def impute(self, dataset:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name}: imputing with Zero.")

        out: DataFrame = dataset.withColumn(
            'y',
            F.coalesce(F.col('y'), F.lit(0.0))
        )

        logger.info(f"Finished: {step_name}")

        return out