from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float64, Int64
from datetime import timedelta
import os

# -------------------------------------------------------------------
# Entity definition
# -------------------------------------------------------------------
transaction = Entity(
    name="transaction_id",
    value_type=ValueType.INT64,
    description="Unique transaction ID",
)

# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------
_current_dir = os.path.dirname(__file__)

parquet_path_v0 = os.path.abspath(
    os.path.join(_current_dir, "../data/v0/transactions_2022.parquet")
)

parquet_path_v1 = os.path.abspath(
    os.path.join(_current_dir, "../data/v1/transactions_2023.parquet")
)

# -------------------------------------------------------------------
# File sources
# -------------------------------------------------------------------
transactions_source_v0 = FileSource(
    path=parquet_path_v0,
   # event_timestamp_column="event_timestamp",
    #created_timestamp_column="created_timestamp",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transactions_source_v1 = FileSource(
    path=parquet_path_v1,
    #event_timestamp_column="event_timestamp",
    #created_timestamp_column="created_timestamp",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# -------------------------------------------------------------------
# Feature schema (IMPORTANT: use Feast types, NOT ValueType)
# -------------------------------------------------------------------
transactions_schema = [
    Field(name="Time", dtype=Float64),
    Field(name="V1", dtype=Float64),
    Field(name="V2", dtype=Float64),
    Field(name="V3", dtype=Float64),
    Field(name="V4", dtype=Float64),
    Field(name="V5", dtype=Float64),
    Field(name="V6", dtype=Float64),
    Field(name="V7", dtype=Float64),
    Field(name="V8", dtype=Float64),
    Field(name="V9", dtype=Float64),
    Field(name="V10", dtype=Float64),
    Field(name="V11", dtype=Float64),
    Field(name="V12", dtype=Float64),
    Field(name="V13", dtype=Float64),
    Field(name="V14", dtype=Float64),
    Field(name="V15", dtype=Float64),
    Field(name="V16", dtype=Float64),
    Field(name="V17", dtype=Float64),
    Field(name="V18", dtype=Float64),
    Field(name="V19", dtype=Float64),
    Field(name="V20", dtype=Float64),
    Field(name="V21", dtype=Float64),
    Field(name="V22", dtype=Float64),
    Field(name="V23", dtype=Float64),
    Field(name="V24", dtype=Float64),
    Field(name="V25", dtype=Float64),
    Field(name="V26", dtype=Float64),
    Field(name="V27", dtype=Float64),
    Field(name="V28", dtype=Float64),
    Field(name="Amount", dtype=Float64),
    Field(name="Class", dtype=Int64),
]

# -------------------------------------------------------------------
# FeatureViews
# -------------------------------------------------------------------
transactions_fv_v0 = FeatureView(
    name="transactions_2022",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=transactions_schema,
    source=transactions_source_v0,
    online=True,
    description="Transactions features for 2022",
)

transactions_fv_v1 = FeatureView(
    name="transactions_2023",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=transactions_schema,
    source=transactions_source_v1,
    online=True,
    description="Transactions features for 2023",
)
