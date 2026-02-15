from __future__ import annotations

import unittest

import pytest
from unittest.mock import MagicMock

pytest.importorskip("pyspark")

from pyspark.sql import DataFrame

from data.business_layer.data_preparation_service.data_preparation_service import DataPreparationService


class TestDataPreparationService(unittest.TestCase):
    def test_data_prepare_executes_steps_in_order(self) -> None:
        raw_df: MagicMock = MagicMock(spec=DataFrame)
        cleaned_df: MagicMock = MagicMock(spec=DataFrame)
        classified_df: MagicMock = MagicMock(spec=DataFrame)
        imputed_df: MagicMock = MagicMock(spec=DataFrame)

        cleaning_pipeline: MagicMock = MagicMock()
        classification_step: MagicMock = MagicMock()
        imputation_pipeline: MagicMock = MagicMock()

        cleaning_pipeline.cleaning.return_value = cleaned_df
        classification_step.classify.return_value = classified_df
        imputation_pipeline.imputation.return_value = imputed_df

        service: DataPreparationService = DataPreparationService(
            data_cleaning_pipeline=cleaning_pipeline,
            classification_step=classification_step,
            segmented_imputation_pipeline=imputation_pipeline,
        )

        result_df: DataFrame = service.data_prepare(raw_dataset=raw_df)

        cleaning_pipeline.cleaning.assert_called_once_with(dataset=raw_df)
        classification_step.classify.assert_called_once_with(dataset=cleaned_df)
        imputation_pipeline.imputation.assert_called_once_with(input_dataset=classified_df)
        self.assertIs(result_df, imputed_df)
