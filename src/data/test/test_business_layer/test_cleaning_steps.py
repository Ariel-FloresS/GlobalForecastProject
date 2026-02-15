from __future__ import annotations

from datetime import date
from typing import List

import pytest

pytest.importorskip("pyspark")

import pyspark.sql.functions as F

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.steps.drop_inactive_recent_series_step import (
    DropInactiveRecentSeriesStep,
)
from data.business_layer.data_cleaning_steps.steps.drop_short_series_step import DropShortSeriesStep
from data.business_layer.data_cleaning_steps.steps.drop_zero_only_series_step import DropZeroOnlySeriesStep
from data.business_layer.data_cleaning_steps.steps.fill_missing_dates_step import FillMissingDatesStep
from data.business_layer.data_cleaning_steps.steps.remove_leading_nulls_steps import RemoveLeadingNullsStep
from data.business_layer.data_cleaning_steps.steps.remove_leading_zeros_step import RemoveLeadingZeroesStep


def _base_dataset(spark_session: SparkSession) -> DataFrame:
    rows: List[tuple[str, str, float | None, str]] = [
        ("id_1", "2024-01-01", None, "A"),
        ("id_1", "2024-01-02", 0.0, "A"),
        ("id_1", "2024-01-03", 3.0, "A"),
        ("id_2", "2024-01-01", 0.0, "B"),
        ("id_2", "2024-01-02", 0.0, "B"),
        ("id_2", "2024-01-03", 0.0, "B"),
    ]
    dataset: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y", "store"])
    output_dataset: DataFrame = dataset.withColumn("ds", F.to_date(F.col("ds")))
    return output_dataset


def test_fill_missing_dates_creates_dense_calendar(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float, str]] = [
        ("id_1", "2024-01-01", 1.0, "A"),
        ("id_1", "2024-01-03", 3.0, "A"),
    ]
    base_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y", "store"])
    input_df: DataFrame = base_df.withColumn("ds", F.to_date(F.col("ds")))

    step: FillMissingDatesStep = FillMissingDatesStep(
        spark=spark_session,
        frequency="D",
        static_features=["store"],
    )

    output_df: DataFrame = step.apply_transformation(input_dataframe=input_df)

    assert output_df.count() == 3
    assert set(output_df.columns) == {"unique_id", "store", "ds", "y"}


def test_remove_leading_nulls_removes_initial_null_records(spark_session: SparkSession) -> None:
    input_df: DataFrame = _base_dataset(spark_session=spark_session)
    step: RemoveLeadingNullsStep = RemoveLeadingNullsStep()

    output_df: DataFrame = step.apply_transformation(input_dataframe=input_df)

    id_1_rows: List[tuple[date, float | None]] = [
        (row["ds"], row["y"])
        for row in output_df.filter("unique_id = 'id_1'").orderBy("ds").select("ds", "y").collect()
    ]
    assert id_1_rows == [(date(2024, 1, 2), 0.0), (date(2024, 1, 3), 3.0)]


def test_remove_leading_zeros_removes_initial_zeros(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float]] = [
        ("id_1", "2024-01-01", 0.0),
        ("id_1", "2024-01-02", 0.0),
        ("id_1", "2024-01-03", 5.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    step: RemoveLeadingZeroesStep = RemoveLeadingZeroesStep()

    output_df: DataFrame = step.apply_transformation(input_dataframe=input_df)

    output_rows: List[tuple[date, float]] = [
        (row["ds"], row["y"]) for row in output_df.orderBy("ds").select("ds", "y").collect()
    ]
    assert output_rows == [(date(2024, 1, 3), 5.0)]


def test_drop_zero_only_series_drops_only_fully_zero_series(spark_session: SparkSession) -> None:
    input_df: DataFrame = _base_dataset(spark_session=spark_session)
    step: DropZeroOnlySeriesStep = DropZeroOnlySeriesStep()

    output_df: DataFrame = step.apply_transformation(input_dataframe=input_df)

    output_ids: List[str] = sorted([row["unique_id"] for row in output_df.select("unique_id").distinct().collect()])
    assert output_ids == ["id_1"]


def test_drop_inactive_recent_series_drops_trailing_zero_streak_series(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float]] = [
        ("id_1", "2024-01-01", 2.0),
        ("id_1", "2024-01-02", 0.0),
        ("id_1", "2024-01-03", 0.0),
        ("id_2", "2024-01-01", 1.0),
        ("id_2", "2024-01-02", 2.0),
        ("id_2", "2024-01-03", 3.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    step: DropInactiveRecentSeriesStep = DropInactiveRecentSeriesStep(inactivity_periods=2)

    output_df: DataFrame = step.apply_transformation(input_dataframe=input_df)
    output_ids: List[str] = [row["unique_id"] for row in output_df.select("unique_id").distinct().collect()]

    assert output_ids == ["id_2"]


def test_drop_short_series_raises_attribute_error_with_short_series(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float]] = [
        ("id_1", "2024-01-01", 1.0),
        ("id_2", "2024-01-01", 1.0),
        ("id_2", "2024-01-02", 2.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    step: DropShortSeriesStep = DropShortSeriesStep(min_records=2)

    with pytest.raises(AttributeError):
        step.apply_transformation(input_dataframe=input_df)
