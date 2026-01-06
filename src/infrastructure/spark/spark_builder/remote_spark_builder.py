from .spark_builder_interface import SparkBuilderInterface
from pyspark.sql import SparkSession
from loguru import logger

class RemoteSparkBuilder(SparkBuilderInterface):

    def build_spark(self)->SparkSession:
        
        spark: SparkSession = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

        logger.info(f'spark remote started.id={id(spark)}')

        return spark


