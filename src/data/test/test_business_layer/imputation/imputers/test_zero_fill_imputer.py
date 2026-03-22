from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.imputation.imputers.zero_fill_imputer import ZeroFillImputer


def test_zero_fill_imputer_replaces_null_with_zero(spark: SparkSession) -> None:
    # Arrange
    input_dataframe: DataFrame = spark.createDataFrame(
        [("s1", datetime.date(2024, 1, 1), None)],
        ["unique_id", "ds", "y"],
    )
    imputer: ZeroFillImputer = ZeroFillImputer()

    # Act
    output_dataframe: DataFrame = imputer.impute(dataset=input_dataframe)

    # Assert
    result_value: float = float(output_dataframe.first()["y"])
    assert result_value == 0.0
