## train_poisoned.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
from constants import EXTERNAL_IP, ML_FLOW_PORT, BUCKET_ID

# ----------------------------
# 0. Set MLflow tracking URI and artifact location
# ----------------------------
mlflow_tracking_uri = f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}"
print(f"[DEBUG] Setting MLflow tracking URI: {mlflow_tracking_uri}")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set artifact location to GCS bucket
mlflow.set_experiment("data_poisoning_experiment")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# ----------------------------
# Poisoning levels to iterate
# ----------------------------
poison_levels = [2, 8, 20]

for level in poison_levels:
    data_path = f"data/v0/poisoned_{level}_percent.csv"
    print(f"[DEBUG] Loading poisoned dataset ({level}%): {data_path}")
    
    df = pd.read_csv(data_path)
    
    # ----------------------------
    # 1. Select numeric features only
    # ----------------------------
    X = df.drop(columns=["Class", "event_timestamp", "created_timestamp"])
    y = df["Class"]
    print(f"[DEBUG] Features shape: {X.shape}, Target shape: {y.shape}")
    
    # ----------------------------
    # 2. Train-validation split
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[DEBUG] Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # ----------------------------
    # 3. Scale features
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ----------------------------
    # 4. Start MLflow run
    # ----------------------------
    run_name = f"poisoned_{level}_percent"
    with mlflow.start_run(run_name=run_name):
        print(f"[DEBUG] Training Logistic Regression on {level}% poisoned data...")
        
        mlflow.log_param("poisoning_level", level)
        
        model = LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs")
        model.fit(X_train_scaled, y_train)
        
        # ----------------------------
        # 5. Predictions & Metrics
        # ----------------------------
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        auc = roc_auc_score(y_val, y_prob)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        print(f"[DEBUG] ROC-AUC Score: {auc}")
        print("[DEBUG] Classification report:\n", classification_report(y_val, y_pred))
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision_class_1", report["1"]["precision"])
        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
        
        # ----------------------------
        # 6. Log classification report as artifact
        # ----------------------------
        mlflow.log_text(
            str(report),
            artifact_file=f"poisoned_{level}/classification_report.txt"
        )

        # ----------------------------
        # 7. Log scaler
        # ----------------------------
        mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path=f"poisoned_{level}_scaler",
            registered_model_name=f"fraud_scaler_poisoned_{level}"
        )
        
        # ----------------------------
        # 8. Log model
        # ----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"poisoned_{level}_fraud_model_lr",
            registered_model_name=f"fraud_detection_lr_poisoned_{level}"
        )
        
        print(f"[DEBUG] Run for {level}% poisoned data complete.\n")

print("[DEBUG] All poisoned data runs complete.")






















# ## train_poisoned.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score
# import mlflow
# import mlflow.sklearn
# from constants import EXTERNAL_IP, ML_FLOW_PORT, BUCKET_ID

# # ----------------------------
# # 0. Set MLflow tracking URI
# # ----------------------------
# mlflow_tracking_uri = f"http://{EXTERNAL_IP}:{ML_FLOW_PORT}"
# print(f"[DEBUG] Setting MLflow tracking URI: {mlflow_tracking_uri}")
# mlflow.set_tracking_uri(mlflow_tracking_uri)

# # ----------------------------
# # Poisoning levels to iterate
# # ----------------------------
# poison_levels = [2, 8, 20]

# for level in poison_levels:
#     data_path = f"data/v0/poisoned_{level}_percent.csv"
#     print(f"[DEBUG] Loading poisoned dataset ({level}%): {data_path}")
    
#     df = pd.read_csv(data_path)
    
#     # ----------------------------
#     # 1. Select numeric features only
#     # ----------------------------
#     # X = df.drop(columns=["Class", "transaction_id", "event_timestamp", "created_timestamp"])
#     X = df.drop(columns=["Class", "event_timestamp", "created_timestamp"])

#     y = df["Class"]
#     print(f"[DEBUG] Features shape: {X.shape}, Target shape: {y.shape}")
    
#     # ----------------------------
#     # 2. Train-validation split
#     # ----------------------------
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )
#     print(f"[DEBUG] Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
#     # ----------------------------
#     # 3. Scale features
#     # ----------------------------
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
    
#     # ----------------------------
#     # 4. MLflow experiment
#     # ----------------------------
#     experiment_name = "data_poisoning_experiment"
#     mlflow.set_experiment(experiment_name)
    
#     with mlflow.start_run(run_name=f"poisoned_{level}_percent"):
#         print(f"[DEBUG] Training Logistic Regression on {level}% poisoned data...")
        
#         mlflow.log_param("poisoning_level", level)
        
#         model = LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs")
#         model.fit(X_train_scaled, y_train)
        
#         # ----------------------------
#         # 5. Predictions & Metrics
#         # ----------------------------
#         y_pred = model.predict(X_val_scaled)
#         y_prob = model.predict_proba(X_val_scaled)[:, 1]
        
#         auc = roc_auc_score(y_val, y_prob)
#         report = classification_report(y_val, y_pred, output_dict=True)
        
#         print(f"[DEBUG] ROC-AUC Score: {auc}")
#         print("[DEBUG] Classification report:\n", classification_report(y_val, y_pred))
        
#         mlflow.log_metric("roc_auc", auc)
#         mlflow.log_metric("precision_class_1", report["1"]["precision"])
#         mlflow.log_metric("recall_class_1", report["1"]["recall"])
#         mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
        
#         # ----------------------------
#         # 6. Log artifacts
#         # ----------------------------
#         # mlflow.log_text(str(report), "classification_report.txt")
#         mlflow.log_text(str(report), f"{BUCKET_ID}/poisoned_{level}/classification_report.txt")

        
#         # ----------------------------
#         # 7. Log scaler
#         # ----------------------------
#         # mlflow.sklearn.log_model(
#         #     sk_model=scaler,
#         #     artifact_path=f"scaler_poisoned_{level}",
#         #     registered_model_name=f"fraud_scaler_poisoned_{level}"
#         # )
#         # mlflow.sklearn.log_model(
#         #     sk_model=scaler,
#         #     artifact_path=f"{BUCKET_ID}/scaler_poisoned_{level}",
#         #     registered_model_name=f"fraud_scaler_poisoned_{level}"
#         # )

#         mlflow.sklearn.log_model(
#             sk_model=scaler,
#             artifact_path=f"{BUCKET_ID}/poisoned_{level}/scaler",
#             registered_model_name=f"fraud_scaler_poisoned_{level}"
#         )
        
#         # ----------------------------
#         # 8. Log model
#         # ----------------------------
#         # mlflow.sklearn.log_model(
#         #     sk_model=model,
#         #     artifact_path=f"fraud_model_lr_poisoned_{level}",
#         #     registered_model_name=f"fraud_detection_lr_poisoned_{level}"
#         # )

#         # mlflow.sklearn.log_model(
#         #     sk_model=model,
#         #     artifact_path=f"{BUCKET_ID}/fraud_model_lr_poisoned_{level}",
#         #     registered_model_name=f"fraud_detection_lr_poisoned_{level}"
#         # )

#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path=f"{BUCKET_ID}/poisoned_{level}/fraud_model_lr",
#             registered_model_name=f"fraud_detection_lr_poisoned_{level}"
#         )

        
#         print(f"[DEBUG] Run for {level}% poisoned data complete.\n")

# print("[DEBUG] All poisoned data runs complete.")
