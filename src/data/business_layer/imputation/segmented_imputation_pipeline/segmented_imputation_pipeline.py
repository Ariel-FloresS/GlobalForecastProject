from .segmented_imputation_pipeline_interface import SegmentedImputationPipelineInterface
from data.business_layer.imputation.imputers import ImputerInterface
from data.business_layer.imputation.config import IMPUTER_BY_CLASS
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from loguru import logger
from functools import reduce
from typing import List, Dict, Set



class SegmentedImputationPipeline(SegmentedImputationPipelineInterface):

    def __init__(self, imputer_by_class: Dict[str, ImputerInterface] = IMPUTER_BY_CLASS)->None:

        self._imputer_by_class = imputer_by_class

        self._known_classes: List[str] = ['Smooth', 'Intermittent', 'Erratic', 'Lumpy']

    def imputation(self, input_dataset: DataFrame)->DataFrame:

        pipeline_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[PIPELINE START]   {pipeline_name}\n{'='*84}"

        logger.info(banner_top)

        required_cols: Set[str] = {"unique_id", "ds", "y", "classification"}
        missing: Set[str] = required_cols.difference(set(input_dataset.columns))

        if missing:
            raise ValueError(
                f"[PIPELINE] {pipeline_name}: missing required columns: {sorted(missing)}. "
                f"Make sure you classified the dataset before imputing."
            )
        
        missing_imputers = [c for c in self._known_classes if c not in self._imputer_by_class.keys()]
        if missing_imputers:
            raise ValueError(
                f"[PIPELINE] {pipeline_name}: No imputer configured for classes: {missing_imputers}. "
                f"Update IMPUTER_BY_CLASS."
            )
        
        
        operate_dataframe: DataFrame = input_dataset.cache()
        imputed_parts: List[DataFrame] = []

        for cls in self._known_classes:

            cls_df: DataFrame = operate_dataframe.filter(F.col("classification") == F.lit(cls))
            imputer: ImputerInterface = self._imputer_by_class[cls]
            imputed_cls_df: DataFrame = imputer.impute(dataset = cls_df)
            imputed_parts.append(imputed_cls_df)

        out: DataFrame = reduce(lambda a, b: a.unionByName(b), imputed_parts)


        banner_bottom: str = f"\n{'='*84}\n[PIPELINE END]   {pipeline_name}\n{'='*84}"
        logger.info(banner_bottom)
        
        return out

        
        
        

