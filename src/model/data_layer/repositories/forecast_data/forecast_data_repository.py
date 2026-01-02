from .forecast_data_repository_interface import ForecastDataRepositoryInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F



class ForecastDataRepository(ForecastDataRepositoryInterface):

    def save_forecast_data(self, forecast_dataframe: DataFrame, delta:str)->None:


        
        save_dataframe: DataFrame = (forecast_dataframe
                                     .withColumn('unique_id', F.col('unique_id').cast('string'))
                                     .withColumn('ds', F.col('ds').cast('date'))
                                     .withColumn("process_date", F.current_timestamp())
                                     )
        
        save_dataframe.write.mode('append').option("mergeSchema", "true").saveAsTable(delta)

