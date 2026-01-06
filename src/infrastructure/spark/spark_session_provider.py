from .spark_session_provider_interface import SparkSessionProviderInterface
from .spark_builder import LocalSparkBuilder, RemoteSparkBuilder
from pyspark.sql import SparkSession
import os

class SparkSessionProvider(SparkSessionProviderInterface):

    

    def get_spark_session(self)->SparkSession:

        if os.getenv('SYSTEM_HOST') == 'local':
            
            return LocalSparkBuilder().build_spark()
        else:
            return RemoteSparkBuilder().build_spark()




def main():

    SparkSessionProvider().get_spark_session()



if __name__ == '__main__':

    main()