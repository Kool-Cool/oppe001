# 🛡️ Fraud Detection ML Service

Your goal is to build a **production-grade machine learning service** that can predict fraudulent transactions in real-time.  
This involves more than just training a model; you must create a robust, scalable, and observable system.

The core task is to build a **binary classification model** that predicts the `Class` column based on the other features.  
You will then expose this model via a REST API, deploy it to a Kubernetes cluster, and implement monitoring, security, and explainability practices.

---

## 📊 Dataset
- **Data Link**: [transactions.csv](https://drive.google.com/file/d/1qAfvEMuhCeuvXcm6l5lJTPefkA2PYtC-/view?usp=drive_link)  
- **Schema**:
  - `Time`: Seconds elapsed since the first transaction  
  - `V1–V28`: Anonymized principal components  
  - `Amount`: Transaction amount  
  - `Class`: Target variable (`1 = fraud`, `0 = non-fraud`)  

---

## 📦 Deliverables

### 1. CI/CD & API Containerization
- **Develop the API**: FastAPI service with `/predict` endpoint (returns prediction + probability score).  
- **Containerize the Service**: Dockerfile packaging FastAPI app, trained model, and dependencies.  
- **Set Up Continuous Deployment**:
  - GitHub Actions workflow (`.github/workflows/cd.yml`) triggered on pushes to `main`.  
  - Builds Docker image, tags, authenticates to Google Cloud, and pushes to Artifact Registry.  
- **Use CML**: Reports build/push status back to the Git commit.  

---

### 2. Deployment, Orchestration & Scaling
- **Deploy to GKE**: Kubernetes manifests (`deployment.yaml`, `service.yaml`) for GKE deployment.  
- **Configure Autoscaling**: HorizontalPodAutoscaler (`hpa.yaml`) scaling based on CPU utilization.  
- **Load Testing**: Locust (`locustfile.py`) simulates concurrent prediction requests.  
- **Observability**: OpenTelemetry instrumentation with custom span for `model.predict()` latency.  

---

### 3. MLSecurityOps: Data Poisoning Attack Simulation
- **Simulate the Attack**:
  - `poisoned_2_percent.csv` → 2% of class `0` flipped to `1`  
  - `poisoned_8_percent.csv` → 8% flipped  
  - `poisoned_20_percent.csv` → 20% flipped  
- **Version the Data**: Track datasets with DVC and push to GCS remote.  
- **Track Experiments**: MLflow logs:
  - Parameter: `poisoning_level` (2, 8, 20)  
  - Metric: F1-score for each run  

---

### 4. Explainability, Fairness & Monitoring
- **Introduce Sensitive Attribute**: Add synthetic `location` column (`Location_A` / `Location_B`).  
- **Explain Predictions**:
  - Train final model (e.g., XGBoost)  
  - Generate SHAP beeswarm plot (`shap_summary.png`) and log to MLflow  
- **Audit for Fairness**:
  - Use Fairlearn to audit location attribute  
  - Log `demographic_parity_difference` metric to MLflow  
- **Detect Concept Drift**:
  - Train on `v0` data, predict on `v1`  
  - Log metrics (F1, precision, recall) for both sets  
  - Save drift comparison plot (`drift_comparison.png`)  

---

## ✅ Summary
This project covers the **end-to-end lifecycle** of a fraud detection ML service:
- Data preparation & versioning  
- CI/CD pipelines & containerization  
- Deployment & autoscaling on GKE  
- Security attack simulation  
- Explainability, fairness, and drift monitoring  

