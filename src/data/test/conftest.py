from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest


PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_PATH: Path = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def spark_session(tmp_path_factory: pytest.TempPathFactory) -> Generator[object, None, None]:
    pyspark_sql = pytest.importorskip("pyspark.sql")
    SparkSession = pyspark_sql.SparkSession

    warehouse_dir: Path = tmp_path_factory.mktemp("spark_warehouse")
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("data-module-unit-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .getOrCreate()
    )
    yield spark
    spark.stop()
