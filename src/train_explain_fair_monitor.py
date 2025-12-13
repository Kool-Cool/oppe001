import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import shap
from fairlearn.metrics import demographic_parity_difference, MetricFrame

from constants import EXTERNAL_IP, ML_FLOW_PORT

# ----------------------------
# 0. Set MLflow tracking URI
# ----------------------------
mlflow_tracking_uri = f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("explainability_fairness_monitoring")

# ----------------------------
# 1. Load clean v0 dataset
# ----------------------------
v0_path = "data/v0/transactions_2022.csv"
v1_path = "data/v1/transactions_2023.csv"
df_v0 = pd.read_csv(v0_path)
df_v1 = pd.read_csv(v1_path)

# ----------------------------
# 2. Introduce synthetic sensitive attribute
# ----------------------------
np.random.seed(42)
df_v0["location"] = np.random.choice(["Location_A", "Location_B"], size=df_v0.shape[0])
df_v1["location"] = np.random.choice(["Location_A", "Location_B"], size=df_v1.shape[0])

# ----------------------------
# 3. Prepare features and target
# ----------------------------
features = [c for c in df_v0.columns if c not in ["Class", "event_timestamp", "created_timestamp"]]
X_v0 = df_v0[features]
y_v0 = df_v0["Class"]

X_v1 = df_v1[features]
y_v1 = df_v1["Class"]

# One-hot encode location
X_v0 = pd.get_dummies(X_v0, columns=["location"], drop_first=True)
X_v1 = pd.get_dummies(X_v1, columns=["location"], drop_first=True)

# ----------------------------
# 4. Train-validation split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_v0, y_v0, test_size=0.2, stratify=y_v0, random_state=42
)

# ----------------------------
# 5. Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_v1_scaled = scaler.transform(X_v1)

# ----------------------------
# 6. Train final model (Logistic Regression)
# ----------------------------
with mlflow.start_run(run_name="logreg_explain_fair_monitor") as run:
    model = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # ----------------------------
    # 7. Predictions
    # ----------------------------
    y_val_pred = model.predict(X_val_scaled)
    y_v1_pred = model.predict(X_v1_scaled)

    f1_val = f1_score(y_val, y_val_pred)
    f1_v1 = f1_score(y_v1, y_v1_pred)

    precision_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)

    precision_v1 = precision_score(y_v1, y_v1_pred)
    recall_v1 = recall_score(y_v1, y_v1_pred)

    # Log metrics
    mlflow.log_metrics({
        "f1_val": f1_val,
        "precision_val": precision_val,
        "recall_val": recall_val,
        "f1_v1": f1_v1,
        "precision_v1": precision_v1,
        "recall_v1": recall_v1
    })

    # ----------------------------
    # 8. SHAP Explainability (fast, new masker API)
    # ----------------------------
    masker = shap.maskers.Independent(X_train_scaled)
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_values = explainer.shap_values(X_val_scaled)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_val, feature_names=X_val.columns, show=False)
    plt.tight_layout()
    shap_plot_file = "shap_summary.png"
    plt.savefig(shap_plot_file)
    mlflow.log_artifact(shap_plot_file)
    plt.close()

    # ----------------------------
    # 9. Fairness audit (demographic parity)
    # ----------------------------
    sensitive_feature = X_val["location_Location_B"] if "location_Location_B" in X_val else np.zeros_like(y_val)
    metric_frame = MetricFrame(metrics=f1_score, y_true=y_val, y_pred=y_val_pred, sensitive_features=sensitive_feature)
    dp_diff = demographic_parity_difference(y_val, y_val_pred, sensitive_features=sensitive_feature)
    mlflow.log_metric("demographic_parity_difference", dp_diff)

    # ----------------------------
    # 10. Concept drift plot
    # ----------------------------
    drift_df = pd.DataFrame({
        "dataset": ["v0_val", "v1_full"],
        "f1_score": [f1_val, f1_v1]
    })
    plt.figure(figsize=(6,4))
    plt.bar(drift_df["dataset"], drift_df["f1_score"], color=["blue", "orange"])
    plt.ylabel("F1 Score")
    plt.title("Concept Drift: v0 vs v1")
    drift_plot_file = "drift_comparison.png"
    plt.savefig(drift_plot_file)
    mlflow.log_artifact(drift_plot_file)
    plt.close()

    # ----------------------------
    # 11. Log scaler and model
    # ----------------------------
    mlflow.sklearn.log_model(scaler, artifact_path="scaler_final", registered_model_name="fraud_scaler_final")
    mlflow.sklearn.log_model(model, artifact_path="logreg_model_final", registered_model_name="fraud_logreg_final")

print("[DEBUG] Explainability, fairness & monitoring run complete.")
