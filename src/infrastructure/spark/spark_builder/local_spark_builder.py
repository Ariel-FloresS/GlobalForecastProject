from .spark_builder_interface import SparkBuilderInterface
from .databricks_connect_config import DatabricksConnectConfig
from databricks.connect import DatabricksSession
from  pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pathlib  import Path
from dotenv import load_dotenv
from loguru import logger
from  typing import Optional
from datetime import datetime
import os

class LocalSparkBuilder(SparkBuilderInterface):

    def __init__(self)->None:

        self._spark: Optional[SparkSession] = None

    def _get_databricks_config(self)->DatabricksConnectConfig:

        project_path: str = Path(__file__).resolve().parents[5]
        config_path: str = 'GlobalForecastProject\src\infrastructure\config\spark\.env'
        load_dotenv(os.path.join(project_path,config_path) )
        cluster_id: str = os.getenv('CLUSTER_ID')
        host: str = os.getenv('HOST')
        token: str = os.getenv('PERSONAL_TOKEN')

        return DatabricksConnectConfig(host = host, token = token, cluster_id = cluster_id)

    def build_spark(self)->SparkSession:

        if self._spark :

            logger.info(f"SPARK SESSION ALREADY ACTIVATED BY: {self._spark.range(1).select(F.current_user()).collect()[0][0]}, ID: {id(self._spark)}")

            return self._spark

        databricks_config: DatabricksConnectConfig = self._get_databricks_config()

        self._spark = (
            DatabricksSession
            .builder
            .host(databricks_config.host)
            .token(databricks_config.token)
            .clusterId(databricks_config.cluster_id)
            .getOrCreate()
            )


        logger.info(f"SPARK SESSION STARTED BY: {self._spark.range(1).select(F.current_user()).collect()[0][0]}, ID: {id(self._spark)}")

        return self._spark

        

