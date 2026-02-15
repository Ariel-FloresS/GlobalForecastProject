from .data_cleaning_step_interface import DataCleaningStepInterface
from pyspark.sql import DataFrame, Row
import pyspark.sql.functions as F
from loguru import logger


class DropShortSeriesStep(DataCleaningStepInterface):

    def __init__(self, min_records: int) -> None:
        self.min_records: int = min_records

    def apply_transformation(self, input_dataframe: DataFrame) -> DataFrame:

        step_name: str = self.__class__.__name__
        logger.info(f"Starting: {step_name} | min_records={self.min_records}")

        if self.min_records <= 0:
            logger.info(f"{step_name}: min_records <= 0, returning original dataframe without changes")
            return input_dataframe

        valid_count_expr: Row = F.sum(
            F.when(
                F.col('y').isNotNull() & ~F.isnan(F.col('y')),
                F.lit(1)
            ).otherwise(F.lit(0))
        ).alias("n_records")

        series_length_df: DataFrame = (
            input_dataframe
            .groupBy("unique_id")
            .agg(valid_count_expr)
        )

        short_series_df: DataFrame = (
            series_length_df
            .filter(F.col("n_records") < F.lit(self.min_records))
            .select("unique_id")
        )

        if short_series_df.limit(1).count() == 0:
            logger.info(f"{step_name}: All series have at least {self.min_records} non-null records")
            return input_dataframe

        dropped_count: int = short_series_df.count()

        output_dataframe: DataFrame = (
            input_dataframe
            .join(short_series_df, on="unique_id", how="left_anti")
        )

        logger.info(
            f"{step_name}: Dropping {dropped_count} time series "
            f"with fewer than {self.min_records} non-null records in {self.target_col}"
        )

        return output_dataframe
