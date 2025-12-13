## test.py


import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn

from constants import EXTERNAL_IP, ML_FLOW_PORT

# ----------------------------
# 0. Set MLflow tracking URI
# ----------------------------
mlflow_tracking_uri = f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}"
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"[DEBUG] MLflow tracking URI set to: {mlflow_tracking_uri}")

# ----------------------------
# 1. Load test data
# ----------------------------
print("\n    Tesing the MODEL on 2023 data\n\n")

df_test = pd.read_csv("data/v1/transactions_2023.csv")
X_test = df_test.drop(columns=["Class", "event_timestamp", "created_timestamp"])
y_test = df_test["Class"]
print(f"[DEBUG] Test data shape: {X_test.shape}, {y_test.shape}")

# ----------------------------
# 2. Load trained model and scaler from MLflow registry
# ----------------------------
model_name = "fraud_detection_lr"
scaler_name = "fraud_scaler"

print(f"[DEBUG] Loading model '{model_name}' from MLflow registry...")
model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

print(f"[DEBUG] Loading scaler '{scaler_name}' from MLflow registry...")
scaler = mlflow.sklearn.load_model(f"models:/{scaler_name}/latest")

# ----------------------------
# 3. Scale features
# ----------------------------
X_test_scaled = scaler.transform(X_test)
print("[DEBUG] Features scaled.")

# ----------------------------
# 4. Make predictions
# ----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("[DEBUG] Predictions completed. Sample probabilities:", y_prob[:5])

# ----------------------------
# 5. Evaluate
# ----------------------------
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------
# 6. Save predictions
# ----------------------------
df_test["predicted_class"] = y_pred
df_test["fraud_probability"] = y_prob
output_path = "data/v1/transactions_2023_predictions.csv"
df_test.to_csv(output_path, index=False)
print(f"[DEBUG] Predictions saved to: {output_path}")
