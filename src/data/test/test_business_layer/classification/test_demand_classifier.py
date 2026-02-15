from __future__ import annotations

import datetime

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.classification.demand_classifier import DemanClassifierFrepple


def test_classify_adds_classification_column(spark: SparkSession) -> None:
    # Arrange
    classifier: DemanClassifierFrepple = DemanClassifierFrepple()
    input_dataframe: DataFrame = spark.createDataFrame(
        [
            ("smooth", datetime.date(2024, 1, 1), 10.0),
            ("smooth", datetime.date(2024, 1, 2), 10.0),
            ("smooth", datetime.date(2024, 1, 3), 10.0),
        ],
        ["unique_id", "ds", "y"],
    )

    # Act
    output_dataframe: DataFrame = classifier.classify(dataset=input_dataframe)

    # Assert
    assert "classification" in output_dataframe.columns
    assert output_dataframe.select("classification").distinct().count() == 1
