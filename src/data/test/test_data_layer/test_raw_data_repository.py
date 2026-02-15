from __future__ import annotations

from datetime import date
from typing import List

import pytest
pytest.importorskip("pyspark")

from pyspark.sql import DataFrame, SparkSession

from data.data_layer.repositories.raw_data.raw_data_repository import RawDataRepository


def test_load_raw_data_standardizes_schema(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float, str]] = [
        ("sku_1", "2024-01-01", 10.0, "store_a"),
        ("sku_1", "2024-01-02", 15.0, "store_a"),
    ]
    input_df: DataFrame = spark_session.createDataFrame(input_rows, ["id", "date", "sales", "store"])
    input_df.createOrReplaceTempView("raw_data_view")

    repository: RawDataRepository = RawDataRepository(spark=spark_session)

    output_df: DataFrame = repository.load_raw_data(
        delta="raw_data_view",
        id_column="id",
        time_column="date",
        target_column="sales",
        static_features=["store"],
    )

    output_rows: List[tuple[str, date, float, str]] = [
        (row["unique_id"], row["ds"], row["y"], row["store"]) for row in output_df.collect()
    ]

    assert output_df.columns == ["unique_id", "ds", "y", "store"]
    assert output_rows == [
        ("sku_1", date(2024, 1, 1), 10.0, "store_a"),
        ("sku_1", date(2024, 1, 2), 15.0, "store_a"),
    ]


def test_load_raw_data_raises_when_delta_is_empty(spark_session: SparkSession) -> None:
    repository: RawDataRepository = RawDataRepository(spark=spark_session)

    with pytest.raises(ValueError, match="delta must be a non-empty string"):
        repository.load_raw_data(
            delta="",
            id_column="id",
            time_column="date",
            target_column="sales",
        )


def test_load_raw_data_raises_on_missing_columns(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str]] = [("sku_1", "2024-01-01")]
    input_df: DataFrame = spark_session.createDataFrame(input_rows, ["id", "date"])
    input_df.createOrReplaceTempView("raw_missing_cols")

    repository: RawDataRepository = RawDataRepository(spark=spark_session)

    with pytest.raises(ValueError, match="Missing required columns"):
        repository.load_raw_data(
            delta="raw_missing_cols",
            id_column="id",
            time_column="date",
            target_column="sales",
        )


def test_train_test_split_returns_non_empty_partitions(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float]] = [
        ("sku_1", "2024-01-01", 2.0),
        ("sku_1", "2024-01-02", 3.0),
        ("sku_1", "2024-01-03", 4.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(input_rows, ["id", "date", "sales"])
    input_df.createOrReplaceTempView("raw_split_ok")

    repository: RawDataRepository = RawDataRepository(spark=spark_session)

    train_df: DataFrame
    test_df: DataFrame
    train_df, test_df = repository.train_test_split(
        split_date="2024-01-02",
        delta="raw_split_ok",
        id_column="id",
        time_column="date",
        target_column="sales",
    )

    assert train_df.count() == 2
    assert test_df.count() == 1


def test_train_test_split_raises_on_empty_partition(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float]] = [
        ("sku_1", "2024-01-01", 2.0),
        ("sku_1", "2024-01-02", 3.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(input_rows, ["id", "date", "sales"])
    input_df.createOrReplaceTempView("raw_split_error")

    repository: RawDataRepository = RawDataRepository(spark=spark_session)

    with pytest.raises(ValueError, match="Split produced empty partition"):
        repository.train_test_split(
            split_date="2024-01-10",
            delta="raw_split_error",
            id_column="id",
            time_column="date",
            target_column="sales",
        )
