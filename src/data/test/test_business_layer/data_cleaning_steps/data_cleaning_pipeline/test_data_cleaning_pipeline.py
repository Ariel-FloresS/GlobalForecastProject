from __future__ import annotations

import datetime
import unittest

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.data_cleaning_pipeline.data_cleaning_pipeline import DataCleaningPipeline
from data.business_layer.data_cleaning_steps.steps.data_cleaning_step_interface import DataCleaningStepInterface
from data.business_layer.data_cleaning_steps.steps.drop_zero_only_series_step import DropZeroOnlySeriesStep


class TestDataCleaningPipeline(unittest.TestCase):
    def test_cleaning_raises_when_step_does_not_implement_interface(self) -> None:
        # Arrange
        pipeline: DataCleaningPipeline = DataCleaningPipeline(cleaning_steps_list=[object()])

        # Act / Assert
        with self.assertRaises(TypeError):
            pipeline.cleaning(dataset=None)  # type: ignore[arg-type]


def test_cleaning_applies_steps_in_sequence(spark: SparkSession) -> None:
    # Arrange
    dataset: DataFrame = spark.createDataFrame(
        [
            ("s1", datetime.date(2024, 1, 1), 0.0),
            ("s1", datetime.date(2024, 1, 2), 0.0),
            ("s2", datetime.date(2024, 1, 1), 1.0),
        ],
        ["unique_id", "ds", "y"],
    )
    step: DataCleaningStepInterface = DropZeroOnlySeriesStep()
    pipeline: DataCleaningPipeline = DataCleaningPipeline(cleaning_steps_list=[step])

    # Act
    output_dataframe: DataFrame = pipeline.cleaning(dataset=dataset)

    # Assert
    ids: list[str] = [row["unique_id"] for row in output_dataframe.select("unique_id").distinct().collect()]
    assert ids == ["s2"]
