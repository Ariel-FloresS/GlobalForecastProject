from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest
from pyspark.sql import SparkSession

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_PATH: Path = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    spark_session: SparkSession = (
        SparkSession.builder.master("local[1]")
        .appName("data-module-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()
