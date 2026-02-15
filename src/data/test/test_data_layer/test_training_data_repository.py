from __future__ import annotations

from datetime import date
from typing import Any, List
from unittest.mock import patch

import pytest
pytest.importorskip("pyspark")

from pyspark.sql import DataFrame, SparkSession

from data.data_layer.repositories.training_data.training_data_repository import TrainingDataRepository


def test_save_training_data_persists_expected_columns(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float, str, float, str]] = [
        ("sku_1", "2024-01-01", 1.0, "Smooth", 0.1, "store_a"),
        ("sku_2", "2024-01-01", 2.0, "Erratic", 0.2, "store_b"),
    ]
    input_df: DataFrame = spark_session.createDataFrame(
        input_rows,
        ["unique_id", "ds", "y", "classification", "exo_1", "store"],
    )

    repository: TrainingDataRepository = TrainingDataRepository()

    with patch("pyspark.sql.readwriter.DataFrameWriter.saveAsTable", autospec=True) as save_mock:
        repository.save_training_data(
            training_dataframe=input_df,
            delta="training_delta_table",
            exogenous_columns=["exo_1"],
            static_features=["store"],
        )

        assert save_mock.call_count == 1
        table_name: Any = save_mock.call_args.args[1]
        assert table_name == "training_delta_table"


def test_save_training_data_replaces_null_targets_before_write(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float | None, str, float]] = [
        ("sku_1", "2024-01-01", None, "Intermittent", 0.1),
    ]
    input_df: DataFrame = spark_session.createDataFrame(
        input_rows,
        ["unique_id", "ds", "y", "classification", "exo_1"],
    )

    repository: TrainingDataRepository = TrainingDataRepository()

    with patch("pyspark.sql.readwriter.DataFrameWriter.saveAsTable", autospec=True):
        repository.save_training_data(
            training_dataframe=input_df,
            delta="training_delta_table",
            exogenous_columns=["exo_1"],
            static_features=None,
        )

    transformed_df: DataFrame = (
        input_df.withColumn("ds", input_df.ds.cast("date")).withColumn("y", input_df.y.cast("double"))
    )
    transformed_rows: List[tuple[str, date, float]] = [
        (row["unique_id"], row["ds"], row["y"] if row["y"] is not None else 0.0)
        for row in transformed_df.fillna({"y": 0.0}).collect()
    ]

    assert transformed_rows == [("sku_1", date(2024, 1, 1), 0.0)]


def test_save_training_data_raises_when_required_columns_are_missing(spark_session: SparkSession) -> None:
    input_rows: List[tuple[str, str, float]] = [("sku_1", "2024-01-01", 5.0)]
    input_df: DataFrame = spark_session.createDataFrame(input_rows, ["unique_id", "ds", "y"])

    repository: TrainingDataRepository = TrainingDataRepository()

    with pytest.raises(ValueError, match="Missing required columns"):
        repository.save_training_data(
            training_dataframe=input_df,
            delta="training_delta_table",
            exogenous_columns=["exo_1"],
            static_features=None,
        )
