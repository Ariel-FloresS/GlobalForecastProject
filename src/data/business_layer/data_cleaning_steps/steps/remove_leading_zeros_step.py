from .data_cleaning_step_interface import DataCleaningStepInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from loguru import logger


class RemoveLeadingZeroesStep(DataCleaningStepInterface):

    def apply_transformation(self, input_dataframe: DataFrame) -> DataFrame:

        step_name: str = self.__class__.__name__
        
        first_nonzero_df: DataFrame = (
            input_dataframe
            .groupBy("unique_id")
            .agg(
                F.min(F.when(F.col("y") != 0, F.col("ds"))).alias("first_nonzero_ds")
            )
        )

        
        output_dataframe: DataFrame = (
            input_dataframe
            .join(first_nonzero_df, on="unique_id", how="left")
            .filter(
                (F.col("first_nonzero_ds").isNull()) |
                (F.col("ds") >= F.col("first_nonzero_ds"))
            )
            .drop("first_nonzero_ds")
        )

        logger.info(f"{step_name}: leading zeros trimmed where applicable.")
        return output_dataframe
