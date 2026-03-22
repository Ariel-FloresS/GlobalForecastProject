from __future__ import annotations

from typing import Tuple

from pyspark.sql import DataFrame, SparkSession

import data.presentation_layer.data_pipeline as data_pipeline_module
from data.presentation_layer.data_pipeline import DataPipeline


class StubRawRepository:
    def __init__(self, spark: SparkSession) -> None:
        self.spark: SparkSession = spark

    def train_test_split(
        self,
        split_date: str,
        delta: str,
        id_column: str,
        time_column: str,
        target_column: str,
        static_features: list[str] | None = None,
    ) -> Tuple[DataFrame, DataFrame]:
        train_df: DataFrame = self.spark.createDataFrame(
            [("a", "2024-01-01", 1.0)], ["unique_id", "ds", "y"]
        )
        test_df: DataFrame = self.spark.createDataFrame(
            [("a", "2024-01-02", 2.0)], ["unique_id", "ds", "y"]
        )
        return train_df, test_df


class StubFeatureStore:
    def __init__(self, spark: SparkSession, frequency: str, season_length: int) -> None:
        self.spark: SparkSession = spark

    def train_dataset(self, historical: DataFrame) -> DataFrame:
        return historical

    def future_dataset(self, historical: DataFrame, horizon: int) -> DataFrame:
        output_df: DataFrame = self.spark.createDataFrame(
            [("a", "2024-01-03", None)], ["unique_id", "ds", "y"]
        )
        return output_df


def test_train_test_future_split_returns_three_dataframes(
    spark: SparkSession, monkeypatch
) -> None:
    # Arrange
    pipeline: DataPipeline = DataPipeline(
        spark=spark,
        raw_delta_table="raw_table",
        id_column="id",
        time_column="date",
        target_column="target",
        frequency="D",
        season_length=7,
        inactivity_periods=2,
        min_records_in_time_series=2,
        exogenous_columns=[],
        training_delta_table="training_table",
    )

    monkeypatch.setattr(data_pipeline_module, "RawDataRepository", StubRawRepository)
    monkeypatch.setattr(data_pipeline_module, "FeatureStore", StubFeatureStore)
    monkeypatch.setattr(
        data_pipeline_module.DataPipeline,
        "_generate_cleaning_dataset",
        lambda self, raw_dataset: raw_dataset,
    )

    # Act
    train_df: DataFrame
    test_df: DataFrame
    future_df: DataFrame
    train_df, test_df, future_df = pipeline.train_test_future_datasets_split(
        split_date="2024-01-01", horizon=1
    )

    # Assert
    assert train_df.count() == 1
    assert test_df.count() == 1
    assert future_df.count() == 1
