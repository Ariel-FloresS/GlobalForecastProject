from __future__ import annotations

from typing import List

import pytest

pytest.importorskip("pyspark")

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from data.business_layer.classification.demand_classifier import DemanClassifierFrepple


def test_demand_classifier_adds_classification_column(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float]] = [
        ("smooth_1", "2024-01-01", 10.0),
        ("smooth_1", "2024-01-02", 11.0),
        ("inter_1", "2024-01-01", 0.0),
        ("inter_1", "2024-01-02", 0.0),
        ("inter_1", "2024-01-03", 5.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    classifier: DemanClassifierFrepple = DemanClassifierFrepple()

    output_df: DataFrame = classifier.classify(dataset=input_df)

    assert "classification" in output_df.columns

    class_map: dict[str, str] = {
        row["unique_id"]: row["classification"]
        for row in output_df.select("unique_id", "classification").distinct().collect()
    }
    assert class_map["smooth_1"] == "Smooth"
    assert class_map["inter_1"] == "Intermittent"


def test_demand_classifier_handles_all_zero_series_as_lumpy(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float]] = [
        ("zero_only", "2024-01-01", 0.0),
        ("zero_only", "2024-01-02", 0.0),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    classifier: DemanClassifierFrepple = DemanClassifierFrepple()

    output_df: DataFrame = classifier.classify(dataset=input_df)

    classification: str = output_df.select("classification").first()["classification"]
    assert classification == "Lumpy"
