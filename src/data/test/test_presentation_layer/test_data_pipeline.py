from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pyspark")

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from data.presentation_layer.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    @staticmethod
    def _pipeline(spark_session: SparkSession) -> DataPipeline:
        pipeline: DataPipeline = DataPipeline(
            spark=spark_session,
            raw_delta_table="raw_table",
            id_column="id",
            time_column="date",
            target_column="sales",
            frequency="D",
            season_length=7,
            inactivity_periods=2,
            min_records_in_time_series=2,
            exogenous_columns=["exo_1"],
            training_delta_table="training_table",
            static_features=["store"],
        )
        return pipeline

    def test_run_pipeline_orchestrates_dependencies(self) -> None:
        spark: SparkSession = (
            SparkSession.builder.master("local[1]").appName("pipeline-test-run").config("spark.sql.shuffle.partitions", "1").getOrCreate()
        )

        raw_df: DataFrame = spark.createDataFrame([("a", "2024-01-01", 1.0)], ["unique_id", "ds", "y"]).withColumn(
            "ds", F.to_date(F.col("ds"))
        )
        clean_df: DataFrame = raw_df
        train_df: DataFrame = spark.createDataFrame(
            [("a", "2024-01-01", 1.0, "Smooth", 0.1, "s1")],
            ["unique_id", "ds", "y", "classification", "exo_1", "store"],
        ).withColumn("ds", F.to_date(F.col("ds")))

        pipeline: DataPipeline = self._pipeline(spark_session=spark)

        with patch("data.presentation_layer.data_pipeline.RawDataRepository") as raw_repo_cls, patch(
            "data.presentation_layer.data_pipeline.FeatureStore"
        ) as feature_store_cls, patch(
            "data.presentation_layer.data_pipeline.TrainingDataRepository"
        ) as training_repo_cls, patch.object(
            DataPipeline, "_generate_cleaning_dataset", return_value=clean_df
        ) as clean_mock:
            raw_repo_instance: MagicMock = raw_repo_cls.return_value
            raw_repo_instance.load_raw_data.return_value = raw_df

            feature_store_instance: MagicMock = feature_store_cls.return_value
            feature_store_instance.train_dataset.return_value = train_df

            training_repo_instance: MagicMock = training_repo_cls.return_value

            pipeline.run_pipeline()

            raw_repo_instance.load_raw_data.assert_called_once()
            clean_mock.assert_called_once_with(raw_dataset=raw_df)
            feature_store_instance.train_dataset.assert_called_once_with(historical=clean_df)
            training_repo_instance.save_training_data.assert_called_once()

        spark.stop()

    def test_train_test_future_datasets_split_returns_expected_tuple(self) -> None:
        spark: SparkSession = (
            SparkSession.builder.master("local[1]").appName("pipeline-test-split").config("spark.sql.shuffle.partitions", "1").getOrCreate()
        )
        base_df: DataFrame = spark.createDataFrame(
            [("a", "2024-01-01", 1.0)],
            ["unique_id", "ds", "y"],
        ).withColumn("ds", F.to_date(F.col("ds")))

        pipeline: DataPipeline = self._pipeline(spark_session=spark)

        with patch("data.presentation_layer.data_pipeline.RawDataRepository") as raw_repo_cls, patch(
            "data.presentation_layer.data_pipeline.FeatureStore"
        ) as feature_store_cls, patch.object(DataPipeline, "_generate_cleaning_dataset", return_value=base_df):
            raw_repo_instance: MagicMock = raw_repo_cls.return_value
            raw_repo_instance.train_test_split.return_value = (base_df, base_df)

            feature_store_instance: MagicMock = feature_store_cls.return_value
            feature_store_instance.train_dataset.return_value = base_df
            feature_store_instance.future_dataset.return_value = base_df

            train_df: DataFrame
            test_df: DataFrame
            future_df: DataFrame
            train_df, test_df, future_df = pipeline.train_test_future_datasets_split(
                split_date="2024-01-01",
                horizon=1,
            )

            self.assertEqual(train_df.count(), 1)
            self.assertEqual(test_df.count(), 1)
            self.assertEqual(future_df.count(), 1)

        spark.stop()
