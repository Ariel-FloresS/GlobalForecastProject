from __future__ import annotations

import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession

from data.business_layer.imputation.imputers.imputer_interface import ImputerInterface
from data.business_layer.imputation.segmented_imputation_pipeline.segmented_imputation_pipeline import SegmentedImputationPipeline


class PassthroughImputer(ImputerInterface):
    def impute(self, dataset: DataFrame) -> DataFrame:
        output_dataframe: DataFrame = dataset.fillna({"y": 0.0})
        return output_dataframe


def test_segmented_imputation_pipeline_processes_all_classes(spark: SparkSession) -> None:
    # Arrange
    dataset: DataFrame = spark.createDataFrame(
        [
            ("a", datetime.date(2024, 1, 1), None, "Smooth"),
            ("b", datetime.date(2024, 1, 1), None, "Intermittent"),
            ("c", datetime.date(2024, 1, 1), None, "Erratic"),
            ("d", datetime.date(2024, 1, 1), None, "Lumpy"),
        ],
        ["unique_id", "ds", "y", "classification"],
    )
    imputer: ImputerInterface = PassthroughImputer()
    pipeline: SegmentedImputationPipeline = SegmentedImputationPipeline(
        imputer_by_class={
            "Smooth": imputer,
            "Intermittent": imputer,
            "Erratic": imputer,
            "Lumpy": imputer,
        }
    )

    # Act
    output_dataframe: DataFrame = pipeline.imputation(input_dataset=dataset)

    # Assert
    assert output_dataframe.count() == 4
    assert output_dataframe.filter("y IS NULL").count() == 0


def test_segmented_imputation_pipeline_raises_when_required_columns_are_missing(spark: SparkSession) -> None:
    # Arrange
    dataset: DataFrame = spark.createDataFrame(
        [("a", datetime.date(2024, 1, 1), 1.0)],
        ["unique_id", "ds", "y"],
    )
    imputer: ImputerInterface = PassthroughImputer()
    pipeline: SegmentedImputationPipeline = SegmentedImputationPipeline(
        imputer_by_class={
            "Smooth": imputer,
            "Intermittent": imputer,
            "Erratic": imputer,
            "Lumpy": imputer,
        }
    )

    # Act / Assert
    with pytest.raises(ValueError):
        pipeline.imputation(input_dataset=dataset)
