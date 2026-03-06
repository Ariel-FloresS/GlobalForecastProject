from .data_cleaning_step_interface import DataCleaningStepInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from loguru import logger


class DropShortSeriesStep(DataCleaningStepInterface):

    def __init__(self, min_records: int) -> None:
        self.min_records: int = min_records

    def apply_transformation(self, input_dataframe: DataFrame) -> DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name} | min_records={self.min_records}")

        if self.min_records <= 0:
            logger.info(
                f"{step_name}: min_records <= 0, returning original dataframe without changes"
            )
            return input_dataframe

        valid_values_df: DataFrame = (
            input_dataframe
            .filter(F.col("y").isNotNull() & ~F.isnan(F.col("y")))
        )

        series_length_df: DataFrame = (
            valid_values_df
            .groupBy("unique_id")
            .agg(F.count(F.lit(1)).alias("n_records"))
        )

        short_series_df: DataFrame = (
            series_length_df
            .filter(F.col("n_records") < F.lit(self.min_records))
            .select("unique_id").distinct()
        )

         
        all_series_df: DataFrame = input_dataframe.select("unique_id").distinct()
        zero_valid_df: DataFrame = all_series_df.join(series_length_df.select("unique_id").distinct(),
                                                      on="unique_id", how="left_anti")

         
        short_series_df: DataFrame = short_series_df.unionByName(zero_valid_df).distinct()

        if short_series_df.limit(1).count() == 0:
            logger.info(
                f"{step_name}: All series have at least {self.min_records} valid records"
            )
            return input_dataframe

        dropped_count: int = short_series_df.select("unique_id").distinct().count()

        output_dataframe: DataFrame = (
            input_dataframe
            .join(short_series_df, on="unique_id", how="left_anti")
        )

        logger.info(
            f"{step_name}: Dropping {dropped_count} time series "
            f"with fewer than {self.min_records} valid records"
        )

        return output_dataframe
