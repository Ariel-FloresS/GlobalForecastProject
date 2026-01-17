from .classifier_interface import ClassifierInterface
from pyspark.sql import DataFrame, Column,Row
import pyspark.sql.functions as F
from loguru import logger
from typing import List, Dict



class DemanClassifierFrepple(ClassifierInterface):

    """
    FrePPLe demand classification using:

    https://frepple.com/blog/demand-classification/

    Thresholds:
      Smooth:       ADI < 1.32 and CV^2 < 0.49
      Intermittent: ADI >= 1.32 and CV^2 < 0.49
      Erratic:      ADI < 1.32 and CV^2 >= 0.49
      Lumpy:        ADI >= 1.32 and CV^2 >= 0.49
    """

    ADI_THR: float = 1.32
    CV2_THR: float = 0.49

    def classify(self, dataset:DataFrame)->DataFrame:

        step_name: str = self.__class__.__name__

        banner_top: str = f"\n{'='*84}\n[CLASSIFICATION START]   {step_name}\n{'='*84}"

        logger.info(banner_top)

        y_nonzero: Column = F.when(F.col('y') > 0, F.col('y'))

        metrics: DataFrame = (
            dataset.groupBy("unique_id")
              .agg(
                  F.count(F.lit(1)).alias("n_periods"),
                  F.sum(F.when(F.col("y") > 0, F.lit(1)).otherwise(F.lit(0))).alias("n_nonzero"),
                  F.avg(y_nonzero).alias("mean_nonzero"),
                  F.stddev_pop(y_nonzero).alias("std_nonzero"),
                )
              .withColumn(
                  "adi",
                  F.when(
                      F.col("n_nonzero") > 0,
                      F.col("n_periods") / F.col("n_nonzero")
                  ).otherwise(F.lit(float("inf")))
              )
              .withColumn(
                  "cv2",
                  F.when(
                      F.col("mean_nonzero").isNotNull() & (F.col("mean_nonzero") > 0),
                      F.pow(F.col("std_nonzero") / F.col("mean_nonzero"), 2)
                  ).otherwise(F.lit(float("inf")))
              )
              .withColumn(
                  "classification",
                  F.when(
                      (F.col("adi") < self.ADI_THR) & (F.col("cv2") < self.CV2_THR),
                      F.lit("Smooth")
                  )
                  .when(
                      (F.col("adi") >= self.ADI_THR) & (F.col("cv2") < self.CV2_THR),
                      F.lit("Intermittent")
                  )
                  .when(
                      (F.col("adi") < self.ADI_THR) & (F.col("cv2") >= self.CV2_THR),
                      F.lit("Erratic")
                  )
                  .otherwise(F.lit("Lumpy"))
              )
              .select("unique_id", "classification")
        )

        out: DataFrame = dataset.join(metrics, on="unique_id", how="left")

        logger.info(f"{step_name}: Finished Added column `classification`.")

        counts_df: DataFrame = (
            out.select("unique_id", "classification")
            .distinct()
            .groupBy("classification")
            .agg(F.count(F.lit(1)).alias("n_series"))
        )
        rows: List[Row] = counts_df.collect()  
        dist: Dict[str, int] = {r["classification"]: int(r["n_series"]) for r in rows}

        ordered_classes: List[str] = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
        dist_str: str = " | ".join([f"{c}={dist.get(c, 0)}" for c in ordered_classes])
        logger.info(f"[CLASSIFICATION] {step_name}: series distribution -> {dist_str}")

        
        banner_bottom: str = f"\n{'='*84}\n[CLASSIFICATION END]   {step_name}\n{'='*84}"
        logger.info(banner_bottom)

        return out