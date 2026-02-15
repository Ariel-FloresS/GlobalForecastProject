from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.imputation.imputers.rolling_median_ffill_imputer import RollingMedianFFillImputer


def test_rolling_median_ffill_imputer_fills_nulls(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("s1", datetime.date(2024, 1, 1), 3.0),
            ("s1", datetime.date(2024, 1, 2), None),
        ],
        ["unique_id", "ds", "y"],
    )
    imputer: RollingMedianFFillImputer = RollingMedianFFillImputer(window_size=2)

    # Act
    output_dataframe: DataFrame = imputer.impute(dataset=input_dataframe)

    # Assert
    assert output_dataframe.filter("y IS NULL").count() == 0
