from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.steps.drop_zero_only_series_step import DropZeroOnlySeriesStep


def test_drop_zero_only_series_removes_all_zero_series(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("drop", datetime.date(2024, 1, 1), 0.0),
            ("keep", datetime.date(2024, 1, 1), 1.0),
        ],
        ["unique_id", "ds", "y"],
    )
    step: DropZeroOnlySeriesStep = DropZeroOnlySeriesStep()

    # Act
    output_dataframe: DataFrame = step.apply_transformation(input_dataframe=input_dataframe)

    # Assert
    ids: list[str] = [row["unique_id"] for row in output_dataframe.select("unique_id").distinct().collect()]
    assert ids == ["keep"]
