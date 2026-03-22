from __future__ import annotations

from unittest.mock import Mock

from pyspark.sql import DataFrame

from data.business_layer.data_preparation_service.data_preparation_service import DataPreparationService


def test_data_prepare_executes_pipeline_in_order() -> None:
    # Arrange
    raw_dataset: DataFrame = Mock(spec=DataFrame)
    cleaning_dataframe: DataFrame = Mock(spec=DataFrame)
    classify_dataframe: DataFrame = Mock(spec=DataFrame)
    imputation_dataframe: DataFrame = Mock(spec=DataFrame)

    cleaning_pipeline: Mock = Mock()
    cleaning_pipeline.cleaning.return_value = cleaning_dataframe
    classification_step: Mock = Mock()
    classification_step.classify.return_value = classify_dataframe
    segmented_imputation_pipeline: Mock = Mock()
    segmented_imputation_pipeline.imputation.return_value = imputation_dataframe

    service: DataPreparationService = DataPreparationService(
        data_cleaning_pipeline=cleaning_pipeline,
        classification_step=classification_step,
        segmented_imputation_pipeline=segmented_imputation_pipeline,
    )

    # Act
    output_dataframe: DataFrame = service.data_prepare(raw_dataset=raw_dataset)

    # Assert
    assert output_dataframe is imputation_dataframe
    cleaning_pipeline.cleaning.assert_called_once_with(dataset=raw_dataset)
    classification_step.classify.assert_called_once_with(dataset=cleaning_dataframe)
    segmented_imputation_pipeline.imputation.assert_called_once_with(input_dataset=classify_dataframe)
