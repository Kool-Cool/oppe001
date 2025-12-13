from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from constants import EXTERNAL_IP, ML_FLOW_PORT

app = FastAPI(title="Fraud Detection API", version="1.0")

# ----------------------------
# Input schema including transaction_id
# ----------------------------
class Transaction(BaseModel):
    transaction_id: int
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# ----------------------------
# Load model and scaler
# ----------------------------
mlflow.set_tracking_uri(f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}")
model = mlflow.sklearn.load_model("models:/fraud_detection_lr/latest")
scaler = mlflow.sklearn.load_model("models:/fraud_scaler/latest")

# ----------------------------
# /predict endpoint (safe version)
# ----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame
    input_df = pd.DataFrame([transaction.dict()])

    # Ensure all scaler features are present
    expected_cols = scaler.feature_names_in_.tolist()
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # fill missing with 0

    # Reorder columns to match scaler
    input_df = input_df[expected_cols]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    pred_class = int(model.predict(input_scaled)[0])
    pred_prob = float(model.predict_proba(input_scaled)[:, 1][0])

    return {
        "transaction_id": transaction.transaction_id,
        "predicted_class": pred_class,
        "fraud_probability": pred_prob
    }

# ----------------------------
# /test endpoint
# ----------------------------
@app.get("/test")
def test_api():
    dummy_transaction = {
        "transaction_id": 123,
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.09,
        "V11": 0.123,
        "V12": 0.456,
        "V13": 0.789,
        "V14": -0.123,
        "V15": 0.234,
        "V16": -0.345,
        "V17": 0.567,
        "V18": -0.678,
        "V19": 0.890,
        "V20": -0.012,
        "V21": 0.345,
        "V22": -0.456,
        "V23": 0.567,
        "V24": -0.678,
        "V25": 0.789,
        "V26": -0.890,
        "V27": 0.123,
        "V28": -0.234,
        "Amount": 100.0
    }
    return {"dummy_transaction": dummy_transaction, "message": "API is working!"}

@app.get("/ping")
def test_ping():
    return {"message": "PONG"}
