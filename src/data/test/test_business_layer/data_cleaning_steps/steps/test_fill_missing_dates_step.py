from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.steps.fill_missing_dates_step import FillMissingDatesStep


def test_fill_missing_dates_generates_dense_calendar(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("s1", datetime.date(2024, 1, 1), 1.0),
            ("s1", datetime.date(2024, 1, 3), 3.0),
        ],
        ["unique_id", "ds", "y"],
    )
    step: FillMissingDatesStep = FillMissingDatesStep(spark=spark, frequency="D")

    # Act
    output_dataframe: DataFrame = step.apply_transformation(input_dataframe=input_dataframe)

    # Assert
    assert output_dataframe.count() == 3
