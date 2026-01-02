from .dataset_partitioning_interface import DatasetPartitioningInterface
from pyspark.sql import DataFrame, SparkSession

class DatasetPartitioning(DatasetPartitioningInterface):

    def __init__(
        self,
        spark: SparkSession,
        multiplier: float = 1.0,
        min_parts: int = 8,
        max_parts: int | None = None,
    ) -> None:
        
        self.spark = spark
        self._multiplier = multiplier
        self._min_parts = min_parts
        self._max_parts = max_parts

    def _auto_partitions(self) -> int:

        base_partition: int = int(self.spark.sparkContext.defaultParallelism * self._multiplier)
        
        parts: int = max(self._min_parts, base_partition)

        return min(parts, self._max_parts) if self._max_parts else parts

    def get_dataset_partition(self, dataset: DataFrame, partition_column: str) -> DataFrame:

        if dataset is None:
            raise ValueError("dataset must not be None.")
        
        if not partition_column or not partition_column.strip():
            raise ValueError("partition_column must be a non-empty string.")
        
        if partition_column not in dataset.columns:
            raise ValueError(
                f"partition_column '{partition_column}' not found in dataset columns: {dataset.columns}"
            )

        partitions = self._auto_partitions()

        return (
                dataset
                .repartitionByRange(partitions, partition_column)
                .sortWithinPartitions(partition_column, 'ds')
            )
