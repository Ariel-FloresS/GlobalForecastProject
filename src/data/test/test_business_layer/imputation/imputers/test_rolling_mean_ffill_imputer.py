from __future__ import annotations

import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.business_layer.imputation.imputers.rolling_mean_ffill_imputer import RollingMeanFFillImputer


def test_rolling_mean_ffill_imputer_fills_nulls(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("s1", datetime.date(2024, 1, 1), 2.0),
            ("s1", datetime.date(2024, 1, 2), None),
        ],
        ["unique_id", "ds", "y"],
    )
    imputer: RollingMeanFFillImputer = RollingMeanFFillImputer(window_size=2)

    # Act
    output_dataframe: DataFrame = imputer.impute(dataset=input_dataframe)

    # Assert
    assert output_dataframe.filter("y IS NULL").count() == 0


def test_rolling_mean_ffill_imputer_raises_when_window_size_invalid(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [("s1", datetime.date(2024, 1, 1), None)],
        ["unique_id", "ds", "y"],
    )
    imputer: RollingMeanFFillImputer = RollingMeanFFillImputer(window_size=0)

    # Act / Assert
    with pytest.raises(ValueError):
        imputer.impute(dataset=input_dataframe)
