from __future__ import annotations

import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.data_layer.repositories.raw_data.raw_data_repository import RawDataRepository


def test_load_raw_data_maps_schema(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("sku-1", datetime.date(2024, 1, 1), 10.0, "A"),
            ("sku-1", datetime.date(2024, 1, 2), 15.0, "A"),
        ],
        ["id", "date", "target", "region"],
    )
    input_dataframe.createOrReplaceTempView("raw_table")
    repository: RawDataRepository = RawDataRepository(spark=spark)

    # Act
    output_dataframe: DataFrame = repository.load_raw_data(
        delta="raw_table",
        id_column="id",
        time_column="date",
        target_column="target",
        static_features=["region"],
    )

    # Assert
    assert output_dataframe.columns == ["unique_id", "ds", "y", "region"]
    assert output_dataframe.count() == 2


def test_train_test_split_returns_non_empty_partitions(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("sku-1", datetime.date(2024, 1, 1), 10.0),
            ("sku-1", datetime.date(2024, 1, 2), 20.0),
            ("sku-1", datetime.date(2024, 1, 3), 30.0),
        ],
        ["id", "date", "target"],
    )
    input_dataframe.createOrReplaceTempView("split_table")
    repository: RawDataRepository = RawDataRepository(spark=spark)

    # Act
    train_dataframe: DataFrame
    test_dataframe: DataFrame
    train_dataframe, test_dataframe = repository.train_test_split(
        split_date="2024-01-02",
        delta="split_table",
        id_column="id",
        time_column="date",
        target_column="target",
    )

    # Assert
    assert train_dataframe.count() == 2
    assert test_dataframe.count() == 1


def test_load_raw_data_raises_when_delta_is_empty(spark: SparkSession) -> None:
    # Arrange
    repository: RawDataRepository = RawDataRepository(spark=spark)

    # Act / Assert
    with pytest.raises(ValueError):
        repository.load_raw_data(
            delta=" ",
            id_column="id",
            time_column="date",
            target_column="target",
        )
