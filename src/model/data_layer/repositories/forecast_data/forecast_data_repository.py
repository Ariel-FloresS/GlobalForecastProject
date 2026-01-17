from .forecast_data_repository_interface import ForecastDataRepositoryInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from loguru import logger


class ForecastDataRepository(ForecastDataRepositoryInterface):

    def save_forecast_data(self, forecast_dataframe:DataFrame, version: str ,delta:str)->None:

        step_name: str = self.__class__.__name__
        
        save_dataframe: DataFrame = (forecast_dataframe
                                     .withColumn('unique_id', F.col('unique_id').cast('string'))
                                     .withColumn('ds', F.col('ds').cast('date'))
                                     .withColumn('y_pred', F.col('y_pred').cast('double'))
                                     .withColumn("process_date", F.current_timestamp())
                                     .withColumn('version', F.lit(version))
                                     )
        
        save_dataframe.write.mode('append').saveAsTable(delta)

        logger.info(f"[{step_name}] Forecast data persisted successfully into '{delta}'.")

        banner_bottom: str = f"\n{'='*84}\n[FORECAST DATA REPOSITORY END]   {step_name}\n{'='*84}"
        logger.info(banner_bottom)

