from .inbound_data_adapter_interface import InboundDataAdapterInterface
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, DoubleType
from typing import List


class InboundDataAdapter(InboundDataAdapterInterface):

    def __init__(self) -> None:
        self._required_columns: List[str] = ['unique_id', 'ds', 'y']

    def inbound_adapter(self, input_dataframe: DataFrame) -> DataFrame:

        missing_required: List[str] = [
            c for c in self._required_columns if c not in input_dataframe.columns
        ]
        
        if missing_required:
            raise ValueError(
                f"Missing required columns {missing_required} in input DataFrame. "
                f"Available columns: {input_dataframe.columns}"
            )

        output_dataframe: DataFrame = (
            input_dataframe
            .withColumn('unique_id', F.col('unique_id').cast(StringType()))
            .withColumn('ds', F.to_date(F.col('ds')))
            .withColumn('y', F.col('y').cast(DoubleType()))
            .dropDuplicates(['unique_id', 'ds'])
        )

        return output_dataframe
