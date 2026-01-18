from .training_data_repository_interface import TrainingDataRepositoryInterface
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from loguru import logger
from typing import List, Optional

class TrainingDataRepository(TrainingDataRepositoryInterface):

    REQUIRED_COLUMNS: List[str] = ['unique_id', 'ds', 'y', 'classification']

    def save_training_data(self, training_dataframe:DataFrame, delta:str, exogenous_columns:List[str], static_features:Optional[List[str]] = None)->None:

        step_name: str = self.__class__.__name__

        if training_dataframe is None:
            raise ValueError(f"[{step_name}] training_dataframe cannot be None.")

        if not delta or not isinstance(delta, str):
            raise ValueError(f"[{step_name}] delta must be a non-empty string.")
        
        static_features: List[str] = static_features or []

        necessary_columns: List[str] = self.REQUIRED_COLUMNS + exogenous_columns + static_features

        missing_required_columns: List[str] = [col for col in necessary_columns if col not in training_dataframe.columns]

        if missing_required_columns:

            raise ValueError(
                f"[{step_name}] Missing required columns for persistence: {missing_required_columns}. "
                f"Available columns: {training_dataframe.columns}"
            )

        has_any_null: bool = (
            training_dataframe
            .filter(F.col('y').isNull())
            .limit(1)
            .count() > 0
        )

        if has_any_null:

            logger.warning(f"[{step_name}] Column 'y' contains NULL values. "
            "Replacing ALL NULLs with 0.0.")

            training_dataframe: DataFrame = training_dataframe.withColumn('y', F.when(F.col('y').isNull(), 0).otherwise(F.col('y')))
        
        save_dataframe: DataFrame = (training_dataframe
                                    .select(necessary_columns)
                                    .withColumn('unique_id', F.col('unique_id').cast('string'))
                                    .withColumn('ds', F.col('ds').cast('date'))
                                    .withColumn('y', F.col('y').cast('double'))
                                    )

        save_dataframe.write.mode('overwrite').saveAsTable(delta)

        logger.info(f"[{step_name}] Training dataset persisted successfully into '{delta}'.")

        banner_bottom: str = f"\n{'='*84}\n[TRAIN DATA REPOSITORY END]   {step_name}\n{'='*84}"
        logger.info(banner_bottom)

        

