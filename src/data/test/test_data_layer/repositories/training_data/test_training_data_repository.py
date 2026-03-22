from __future__ import annotations

import datetime
import uuid

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.data_layer.repositories.training_data.training_data_repository import TrainingDataRepository


def test_save_training_data_persists_required_columns(spark: SparkSession) -> None:
    # Arrange
    repository: TrainingDataRepository = TrainingDataRepository()
    table_name: str = f"training_table_{uuid.uuid4().hex[:8]}"
    training_dataframe: DataFrame = spark.createDataFrame(
        [("sku-1", datetime.date(2024, 1, 1), 1.0, "Smooth", 9.0)],
        ["unique_id", "ds", "y", "classification", "exo_1"],
    )

    # Act
    repository.save_training_data(
        training_dataframe=training_dataframe,
        delta=table_name,
        exogenous_columns=["exo_1"],
    )
    saved_dataframe: DataFrame = spark.table(table_name)

    # Assert
    assert saved_dataframe.columns == ["unique_id", "ds", "y", "classification", "exo_1"]
    assert saved_dataframe.count() == 1
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def test_save_training_data_raises_on_missing_columns(spark: SparkSession) -> None:
    # Arrange
    repository: TrainingDataRepository = TrainingDataRepository()
    training_dataframe: DataFrame = spark.createDataFrame(
        [("sku-1", datetime.date(2024, 1, 1), 1.0)],
        ["unique_id", "ds", "y"],
    )

    # Act / Assert
    with pytest.raises(ValueError):
        repository.save_training_data(
            training_dataframe=training_dataframe,
            delta="table_missing_columns",
            exogenous_columns=[],
        )
