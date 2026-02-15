from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import pytest
pytest.importorskip("pyspark")

from pyspark.sql import DataFrame

from data.business_layer.data_cleaning_steps.data_cleaning_pipeline.data_cleaning_pipeline import DataCleaningPipeline
from data.business_layer.data_cleaning_steps.steps.data_cleaning_step_interface import DataCleaningStepInterface


class DummyCleaningStep(DataCleaningStepInterface):
    def __init__(self, out_df: DataFrame) -> None:
        self.out_df: DataFrame = out_df

    def apply_transformation(self, input_dataframe: DataFrame) -> DataFrame:
        return self.out_df


class TestDataCleaningPipeline(unittest.TestCase):
    def test_returns_original_dataset_when_no_steps(self) -> None:
        input_df: MagicMock = MagicMock(spec=DataFrame)
        pipeline: DataCleaningPipeline = DataCleaningPipeline(cleaning_steps_list=[])

        output_df: DataFrame = pipeline.cleaning(dataset=input_df)

        self.assertIs(output_df, input_df)


@pytest.mark.parametrize("invalid_step", [object(), "bad_step", 1])
def test_raises_type_error_for_invalid_step(invalid_step: object) -> None:
    input_df: MagicMock = MagicMock(spec=DataFrame)
    pipeline: DataCleaningPipeline = DataCleaningPipeline(cleaning_steps_list=[invalid_step])

    with pytest.raises(TypeError, match="must implement DataCleaningStepInterface"):
        pipeline.cleaning(dataset=input_df)


def test_applies_all_cleaning_steps_sequentially() -> None:
    df_1: MagicMock = MagicMock(spec=DataFrame)
    df_2: MagicMock = MagicMock(spec=DataFrame)
    df_3: MagicMock = MagicMock(spec=DataFrame)

    step_1: MagicMock = MagicMock(spec=DataCleaningStepInterface)
    step_2: MagicMock = MagicMock(spec=DataCleaningStepInterface)

    step_1.apply_transformation.return_value = df_2
    step_2.apply_transformation.return_value = df_3

    pipeline: DataCleaningPipeline = DataCleaningPipeline(cleaning_steps_list=[step_1, step_2])

    result_df: DataFrame = pipeline.cleaning(dataset=df_1)

    step_1.apply_transformation.assert_called_once_with(input_dataframe=df_1)
    step_2.apply_transformation.assert_called_once_with(input_dataframe=df_2)
    assert result_df is df_3
