# Fraud Detection MLOps on GKE 🚀

**real-time fraud detection system** for online payment transactions.  
This project demonstrates the full lifecycle of an ML system: from data preparation and model training to containerization, deployment, monitoring, and security.

---

## 📂 Repository Structure
- **data/** → Versioned datasets (`v0`, `v1`, poisoned variants) tracked with DVC  
- **src/** → Core FastAPI service and model code  
- **feature_repo/** → Feast feature store setup  
- **.github/workflows/** → CI/CD pipelines with GitHub Actions + CML reporting  
- **deployment.yaml, service.yaml, hpa.yaml** → Kubernetes manifests for GKE deployment and autoscaling  
- **locustfile.py** → Load testing scripts  
- **Dockerfile** → Containerization of FastAPI + model  
- **requirements-*.txt** → Environment dependencies for API, training, CI/CD  

---

## 📊 Dataset
- **Source**: European cardholder transactions (`transactions.csv`)  
- **Schema**: 31 numerical columns (Time, V1–V28, Amount, Class)  
- **Class**: `1 = fraud`, `0 = non-fraud`  
- **Preparation**:
  - Split into `data/v0/transactions_2022.csv` and `data/v1/transactions_2023.csv`  
  - Poisoned datasets created with flipped labels (2%, 8%, 20%)  

---

## 🛠 Features
### 1. CI/CD & Containerization
- FastAPI `/predict` endpoint (returns fraud probability + prediction)  
- Dockerized service pushed to Google Artifact Registry  
- GitHub Actions workflow with CML reporting  

### 2. Deployment & Scaling
- GKE deployment with LoadBalancer service  
- HorizontalPodAutoscaler (HPA) for CPU-based scaling  
- Load testing with Locust  

### 3. MLSecurityOps
- Data poisoning attack simulation  
- DVC for dataset versioning  
- MLflow experiment tracking with poisoning-level parameter  

### 4. Explainability & Fairness
- SHAP beeswarm plots for feature importance  
- Fairlearn audits for demographic parity difference  
- Concept drift detection between v0 and v1 datasets  

---

## 📈 Monitoring & Observability
- **OpenTelemetry** instrumentation for API latency and model inference time  
- **MLflow** logging of metrics, parameters, and artifacts  
- **Drift comparison plots** for v0 vs v1 performance  

---

