from __future__ import annotations

import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.steps.drop_short_series_step import DropShortSeriesStep


def test_drop_short_series_keeps_when_threshold_met(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [("keep", datetime.date(2024, 1, 1), 1.0), ("keep", datetime.date(2024, 1, 2), 2.0)],
        ["unique_id", "ds", "y"],
    )
    step: DropShortSeriesStep = DropShortSeriesStep(min_records=2)

    # Act
    output_dataframe: DataFrame = step.apply_transformation(input_dataframe=input_dataframe)

    # Assert
    assert output_dataframe.count() == 2


def test_drop_short_series_raises_attribute_error_when_dropping(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [("drop", datetime.date(2024, 1, 1), 1.0)],
        ["unique_id", "ds", "y"],
    )
    step: DropShortSeriesStep = DropShortSeriesStep(min_records=2)

    # Act / Assert
    with pytest.raises(AttributeError):
        step.apply_transformation(input_dataframe=input_dataframe)
