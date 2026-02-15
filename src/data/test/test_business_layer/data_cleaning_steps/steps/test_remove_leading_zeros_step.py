from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.data_cleaning_steps.steps.remove_leading_zeros_step import RemoveLeadingZeroesStep


def test_remove_leading_zeros_trims_prefix(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("s1", datetime.date(2024, 1, 1), 0.0),
            ("s1", datetime.date(2024, 1, 2), 3.0),
        ],
        ["unique_id", "ds", "y"],
    )
    step: RemoveLeadingZeroesStep = RemoveLeadingZeroesStep()

    # Act
    output_dataframe: DataFrame = step.apply_transformation(input_dataframe=input_dataframe)

    # Assert
    assert output_dataframe.count() == 1
