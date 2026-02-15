from __future__ import annotations

from typing import Dict, List

import pytest

pytest.importorskip("pyspark")

import pyspark.sql.functions as F

from pyspark.sql import DataFrame, SparkSession

from data.business_layer.imputation.imputers.imputer_interface import ImputerInterface
from data.business_layer.imputation.imputers.rolling_mean_ffill_imputer import RollingMeanFFillImputer
from data.business_layer.imputation.imputers.rolling_median_ffill_imputer import RollingMedianFFillImputer
from data.business_layer.imputation.imputers.zero_fill_imputer import ZeroFillImputer
from data.business_layer.imputation.segmented_imputation_pipeline.segmented_imputation_pipeline import (
    SegmentedImputationPipeline,
)


class _ConstantImputer(ImputerInterface):
    def __init__(self, value: float) -> None:
        self.value: float = value

    def impute(self, dataset: DataFrame) -> DataFrame:
        output_df: DataFrame = dataset.withColumn("y", F.coalesce(F.col("y"), F.lit(self.value)))
        return output_df


def _imputation_input(spark_session: SparkSession) -> DataFrame:
    rows: List[tuple[str, str, float | None, str]] = [
        ("id_s", "2024-01-01", None, "Smooth"),
        ("id_i", "2024-01-01", None, "Intermittent"),
        ("id_e", "2024-01-01", None, "Erratic"),
        ("id_l", "2024-01-01", None, "Lumpy"),
    ]
    base_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y", "classification"])
    output_df: DataFrame = base_df.withColumn("ds", F.to_date(F.col("ds")))
    return output_df


def test_zero_fill_imputer_replaces_null_with_zero(spark_session: SparkSession) -> None:
    input_df: DataFrame = _imputation_input(spark_session=spark_session).filter(F.col("classification") == "Intermittent")
    imputer: ZeroFillImputer = ZeroFillImputer()

    output_df: DataFrame = imputer.impute(dataset=input_df)

    assert output_df.select("y").first()["y"] == 0.0


def test_rolling_mean_ffill_imputer_replaces_null(spark_session: SparkSession) -> None:
    rows: List[tuple[str, str, float | None]] = [
        ("id_1", "2024-01-01", 2.0),
        ("id_1", "2024-01-02", None),
    ]
    input_df: DataFrame = spark_session.createDataFrame(rows, ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )
    imputer: RollingMeanFFillImputer = RollingMeanFFillImputer(window_size=2)

    output_df: DataFrame = imputer.impute(dataset=input_df).orderBy("ds")

    values: List[float] = [row["y"] for row in output_df.select("y").collect()]
    assert values == [2.0, 2.0]


def test_rolling_median_ffill_imputer_validates_window_size(spark_session: SparkSession) -> None:
    input_df: DataFrame = spark_session.createDataFrame([("id_1", "2024-01-01", None)], ["unique_id", "ds", "y"]).withColumn(
        "ds", F.to_date(F.col("ds"))
    )

    with pytest.raises(ValueError, match="window_size must be >= 1"):
        RollingMedianFFillImputer(window_size=0).impute(dataset=input_df)


def test_segmented_imputation_pipeline_routes_by_class(spark_session: SparkSession) -> None:
    input_df: DataFrame = _imputation_input(spark_session=spark_session)
    imputer_by_class: Dict[str, ImputerInterface] = {
        "Smooth": _ConstantImputer(1.0),
        "Intermittent": _ConstantImputer(2.0),
        "Erratic": _ConstantImputer(3.0),
        "Lumpy": _ConstantImputer(4.0),
    }
    pipeline: SegmentedImputationPipeline = SegmentedImputationPipeline(imputer_by_class=imputer_by_class)

    output_df: DataFrame = pipeline.imputation(input_dataset=input_df)

    values_by_id: Dict[str, float] = {row["unique_id"]: row["y"] for row in output_df.select("unique_id", "y").collect()}
    assert values_by_id == {"id_s": 1.0, "id_i": 2.0, "id_e": 3.0, "id_l": 4.0}


def test_segmented_imputation_pipeline_validates_required_columns(spark_session: SparkSession) -> None:
    bad_df: DataFrame = spark_session.createDataFrame([("id_1", 1.0)], ["unique_id", "y"])
    pipeline: SegmentedImputationPipeline = SegmentedImputationPipeline()

    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.imputation(input_dataset=bad_df)
