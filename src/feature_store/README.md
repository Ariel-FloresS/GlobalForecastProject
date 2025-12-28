# Feature Store – Global Forecast Project

**Author & Credits:**  
All credits for the design and implementation of this Feature Store belong to  
**Ariel Flores Santacruz**.
---

This module implements a **time-series–oriented Feature Store** designed to generate and manage
**exogenous variables** in a scalable and reusable way using **Spark + pandas (`applyInPandas`)**.
It follows a clear **layered architecture** that separates orchestration, business logic,
and execution concerns.

---

## Project Structure
```python
feature_store/
├── __init__.py
├── presentation_layer/
│   ├── __init__.py
│   ├── feature_store.py
│   └── feature_store_interface.py
│
├── business_layer/
│   ├── __init__.py
│   ├── exogenous/
│   │   ├── __init__.py
│   │   ├── exogenous_variable_interface.py
│   │   └── mstl_decomposition.py
│   │
│   ├── feature_service/
│   │   ├── __init__.py
│   │   ├── feature_service.py
│   │   └── feature_service_interface.py
│   │
│   └── future_dataset/
│       ├── __init__.py
│       ├── generate_future_dataset.py
│       └── generate_future_dataset_interface.py
│
└── data_layer/
    ├── __init__.py
    ├── inbound/
    │   ├── __init__.py
    │   └── data_adapter/
    │       ├── __init__.py
    │       ├── inbound_data_adapter.py
    │       └── inbound_data_adapter_interface.py
    │
    └── pandas_executor_in_spark/
        ├── __init__.py
        ├── pandas_executor_in_spark_per_time_series.py
        └── pandas_executor_in_spark_per_time_series_interface.py
```
---

## Architecture Overview

The Feature Store is organized into **three layers**, with a strict dependency flow:

Presentation Layer  
↓  
Business Layer  
↓  
Data Layer  

Each layer has a single responsibility and depends only on the layer below it.

---

## Presentation Layer

**Purpose:** Public API and orchestration layer.

- Exposes the `FeatureStore` interface.
- Coordinates data validation, future dataset generation, and feature computation.
- Hides internal implementation details from the user.

This layer contains no business logic and no low-level Spark or pandas execution.

---

## Business Layer

**Purpose:** Core business logic for feature computation.

### Exogenous Variables
- Each exogenous feature is implemented as an independent strategy.
- All features implement `ExogenousVariableInterface`.
- Examples: MSTL decomposition, future Fourier terms, etc.
- Features are plug-and-play and reusable across training and inference.

### Feature Service
- Orchestrates the execution of multiple exogenous variables.
- Applies features sequentially while preserving data consistency.

### Future Dataset
- Generates the future time index based on `horizon` and `frequency`.
- Produces a base future DataFrame without the target variable.

---

## Data Layer

**Purpose:** Technical execution and data adaptation.

### Inbound Data Adapter
- Validates required columns (`unique_id`, `ds`, `y`).
- Casts data types to a standardized schema.
- Ensures clean and consistent input before feature computation.

### Pandas Executor in Spark
- Executes pandas-based logic per time series using `groupBy().applyInPandas`.
- Encapsulates Spark execution and serialization details.
- Enables features that are not natively supported in Spark.

---

## Execution Flow

1. User calls the `FeatureStore`.
2. Input data is validated and standardized.
3. A future dataset is generated (if required).
4. Exogenous variables are applied sequentially.
5. A DataFrame ready for modeling or inference is returned.

---

## Example Usage

```python

from feature_store.presentation_layer import FeatureStoreInterface, FeatureStore

from pyspark.sql import DataFrame, SparkSession


spark_session: SparkSession = SparkSession.builder.appName("GlobalForecast-FeatureStore").getOrCreate()

data: DataFrame = spark_session.sql('select * from catalog.schema.historical_table')


feature_store = FeatureStore(
    spark = spark_session,
    frequency ="M",
    season_length=12
)
train_dataset =  feature_store.train_dataset(historical = data)

future_dataset = feature_store.future_dataset(
    historical = data,
    horizon = 3
)
