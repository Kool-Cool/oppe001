import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
from constants import BUCKET_ID, EXTERNAL_IP, ML_FLOW_PORT

# ----------------------------
# 0. Set MLflow tracking URI
# ----------------------------
mlflow_tracking_uri = f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}"
print(f"[DEBUG] Setting MLflow tracking URI: {mlflow_tracking_uri}")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# ----------------------------
# 1. Load data
# ----------------------------
data_path = "data/v0/transactions_2022.csv"
print(f"[DEBUG] Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"[DEBUG] Data shape: {df.shape}")

# ----------------------------
# 2. Features and target
# ----------------------------
X = df.drop(columns=["Class", "event_timestamp", "created_timestamp"])
y = df["Class"]
print(f"[DEBUG] Features shape: {X.shape}, Target shape: {y.shape}")

# ----------------------------
# 3. Train-validation split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[DEBUG] Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# ----------------------------
# 4. Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print(f"[DEBUG] Features scaled. Sample of scaled data:\n{X_train_scaled[:2]}")

# ----------------------------
# 5. Set MLflow experiment
# ----------------------------
experiment_name = "fraud_detection_lr"
print(f"[DEBUG] Setting MLflow experiment: {experiment_name}")
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="logistic_regression"):

    print("[DEBUG] Training Logistic Regression model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs"
    )
    model.fit(X_train_scaled, y_train)
    print("[DEBUG] Model training complete.")

    # ----------------------------
    # 7. Predictions
    # ----------------------------
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    print("[DEBUG] Predictions complete. Sample probabilities:", y_prob[:5])

    # ----------------------------
    # 8. Metrics
    # ----------------------------
    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, output_dict=True)
    print(f"[DEBUG] ROC-AUC Score: {auc}")
    print("[DEBUG] Classification report:\n", classification_report(y_val, y_pred))

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision_class_1", report["1"]["precision"])
    mlflow.log_metric("recall_class_1", report["1"]["recall"])
    mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
    print("[DEBUG] Metrics logged to MLflow.")

    # ----------------------------
    # 9. Log artifacts
    # ----------------------------
    mlflow.log_text(str(report), "classification_report.txt")
    print("[DEBUG] Classification report logged as artifact.")

    # ----------------------------
    # 10. Log scaler
    # ----------------------------
    mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path="scaler",
        registered_model_name="fraud_scaler"
    )
    print("[DEBUG] Scaler model logged to MLflow and registered.")

    # ----------------------------
    # 11. Log and register model
    # ----------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="fraud_model_lr",
        registered_model_name="fraud_detection_lr"
    )
    print("[DEBUG] Logistic Regression model logged to MLflow and registered.")

print("[DEBUG] MLflow run complete.")
