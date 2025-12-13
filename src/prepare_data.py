import pandas as pd
import os

# Make sure local folders exist
os.makedirs("data/v0", exist_ok=True)
os.makedirs("data/v1", exist_ok=True)

bucket = "gs://oppe2_pract"

# Read CSV from GCS (requires gcsfs and authentication)
df = pd.read_csv(f"{bucket}/transactions.csv")

# Sort and split
df_sorted = df.sort_values("Time").reset_index(drop=True)
mid = len(df_sorted) // 2

df_22 = df_sorted.iloc[:mid].copy()
df_23 = df_sorted.iloc[mid:].copy()

# Add timestamps
start_22 = pd.Timestamp("2022-01-01")
df_22["event_timestamp"] = start_22 + pd.to_timedelta(df_22["Time"], unit="s")
df_22["created_timestamp"] = df_22["event_timestamp"]

start_23 = pd.Timestamp("2023-01-01")
df_23["event_timestamp"] = start_23 + pd.to_timedelta(df_23["Time"], unit="s")
df_23["created_timestamp"] = df_23["event_timestamp"]



## No explicit user_id or entity key:
## This transaction_id can serve as the Feast entity.
df_22["transaction_id"] = range(len(df_22))
df_23["transaction_id"] = range(len(df_23))


# Save locally for DVC tracking
df_22.to_csv("data/v0/transactions_2022.csv", index=False)
df_23.to_csv("data/v1/transactions_2023.csv", index=False)


# Save Parquet for Feast
df_22.to_parquet("data/v0/transactions_2022.parquet", index=False)
df_23.to_parquet("data/v1/transactions_2023.parquet", index=False)



print(df_22.head())
print(df_23.head())
