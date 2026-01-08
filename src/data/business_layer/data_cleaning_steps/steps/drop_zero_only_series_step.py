from business_layer.data_cleaning_steps import DataCleaningStepInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from loguru import logger


class DropZeroOnlySeriesStep(DataCleaningStepInterface):

    def apply_transformation(self, input_dataframe:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        logger.info(f"Starting: {step_name}")

        ids_zero_only_series: DataFrame = (
            input_dataframe
            .groupBy('unique_id')
            .agg(F.max('y').alias('max_y'))
            .filter(F.col('max_y') == 0)
            .select('unique_id')
        )

        if ids_zero_only_series.rdd.isEmpty():

            logger.info(f"{step_name}: No zero-only series found. Skipping.")

            return input_dataframe
        
        output_dataframe: DataFrame = (input_dataframe
                                       .join(other = ids_zero_only_series, on = 'unique_id', how = 'left_anti')
                                       )
        
        logger.info(f"{step_name}: Zero-only series found. Dropping {ids_zero_only_series.count()} time series.")

        return output_dataframe
        