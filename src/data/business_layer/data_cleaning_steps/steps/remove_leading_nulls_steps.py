from .data_cleaning_step_interface import DataCleaningStepInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from loguru import logger


class RemoveLeadingNullsStep(DataCleaningStepInterface):

    def apply_transformation(self, input_dataframe: DataFrame) -> DataFrame:

        step_name: str = self.__class__.__name__
        
        first_nonnull_df: DataFrame = (
            input_dataframe
            .groupBy("unique_id")
            .agg(
                F.min(F.when(F.col("y").isNotNull(), F.col("ds"))).alias("first_nonnull_ds")
            )
        )

        output_dataframe: DataFrame = (
            input_dataframe
            .join(first_nonnull_df, on="unique_id", how="left")
            .filter(
                (F.col("first_nonnull_ds").isNull()) |
                (F.col("ds") >= F.col("first_nonnull_ds"))
            )
            .drop("first_nonnull_ds")
        )

        logger.info(
            f"{step_name}: leading null values trimmed where applicable."
        )

        
        return output_dataframe
