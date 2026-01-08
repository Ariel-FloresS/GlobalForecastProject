from business_layer.data_cleaning_steps import DataCleaningStepInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql import Window
from loguru import logger


class DropInactiveRecentSeriesStep(DataCleaningStepInterface):

    def __init__(self, inactivity_periods:int)->None:

        self.inactivity_periods = inactivity_periods

    def apply_transformation(self, input_dataframe:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name} | inactivity_periods={self.inactivity_periods}")

        if self.inactivity_periods <= 0:

            logger.info(f"{step_name}: inactivity_periods <= 0, returning original dataframe")

            return input_dataframe
        
        w_order: Window = Window.partitionBy("unique_id").orderBy(F.col("ds").asc())

        zero_marked_df: DataFrame = (
            input_dataframe
            .withColumn("is_zero", F.when(F.col("y") == 0, 1).otherwise(0))
            .withColumn(
                "zero_group",
                F.sum(F.when(F.col("is_zero") == 0, 1).otherwise(0)).over(w_order)
            )
        )

        w_group: Window = Window.partitionBy("unique_id", "zero_group")

        streak_df: DataFrame = (
            zero_marked_df
            .withColumn("zero_streak", F.sum(F.col("is_zero")).over(w_group).cast("int"))
        )

        w_recent: Window = Window.partitionBy("unique_id").orderBy(F.col("ds").desc())

        last_row_df: DataFrame = (
            streak_df
            .withColumn("rn", F.row_number().over(w_recent).cast("int"))
            .filter(F.col("rn") == F.lit(1))
            .select("unique_id", "zero_streak", "is_zero")
        )

        inactive_ids_df: DataFrame = (
            last_row_df
            .filter((F.col("is_zero") == 1) & (F.col("zero_streak") >= self.inactivity_periods))
            .select("unique_id")
            .distinct()
        )

        if inactive_ids_df.rdd.isEmpty():
            logger.info(f"{step_name}: No series with >= {self.inactivity_periods} trailing zeros found, returning original dataframe")
            return input_dataframe

        
        output_df: DataFrame = input_dataframe.join(inactive_ids_df, on="unique_id", how="left_anti")

        logger.info(f"{step_name}: inactivity series found. Dropping {inactive_ids_df.count()} time series.")

        return output_df
        

        
