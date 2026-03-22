from __future__ import annotations

from pathlib import Path
import unittest

from pyspark.sql import SparkSession


class DataLayerTestOrchestrator:
    def __init__(self, spark: SparkSession) -> None:
        self.spark: SparkSession = spark

    def run(self) -> unittest.result.TestResult:
        root_path: Path = Path(__file__).resolve().parent
        suite: unittest.TestSuite = unittest.defaultTestLoader.discover(str(root_path), pattern="test_*.py")
        runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=2)
        result: unittest.result.TestResult = runner.run(suite)
        return result
