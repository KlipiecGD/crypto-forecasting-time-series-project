# 🪙 Bitcoin Volatility Predictor

## 📌 General Description

The **Bitcoin Volatility Predictor** is a Machine Learning application that forecasts Bitcoin volatility 1–30 days into the future, classifying market conditions into **Low**, **Normal**, or **High** volatility states to help traders and analysts make informed decisions.

The project supports two deployment modes:
- **Local**: Run inference entirely on your machine
- **AWS**: Model served via SageMaker, UI hosted on Elastic Beanstalk with CloudWatch logging

---

## 📂 Repository Structure

```text
.
├── data/                               # Local data storage (CSVs)
├── models/                             # Saved model artifacts
├── notebooks/                          # Jupyter notebooks for experimentation
├── src/
│   ├── config/                         # YAML config + Python config class
│   ├── dags/                           # Airflow DAGs
│   ├── elastic_beanstalk/              # EB deployment & teardown scripts
│   ├── features/                       # Feature engineering (DataTransformer)
│   ├── fetch_data/                     # Binance API + Kaggle data fetchers
│   ├── inference/                      # Inference helpers (data loader, predictor)
│   ├── logging_utils/                  # CloudWatch-enabled loggers
│   ├── monitoring/                     # Evidently drift + performance monitoring
│   ├── pipelines/                      # Training & inference pipelines
│   ├── sagemaker_deployment/           # Docker/ECR/SageMaker deployment
│   ├── training/                       # Model training, evaluation, saving
│   └── ui/                             # Streamlit components + data utilities
├── tests/                              # Unit tests for data processing 
├── .dockerignore                       # Exclude files from Docker context
├── .gitignore                          # Exclude files from Git
├── cleanup_resources.py                # Terminate EB + SageMaker in one command
├── main_aws.py                         # Streamlit app (AWS / SageMaker inference)
├── main_local.py                       # Streamlit app (local inference)
├── main_orchestrator.py                # Full end-to-end deployment orchestrator
├── Dockerfile.streamlit                # Docker image for Streamlit on EB
├── airflow.env                         # Airflow environment variables
├── requirements.streamlit.txt          # Pinned deps for the Streamlit container
├── requirements.txt                    # Full project dependencies
├── deployment_info.json                # SageMaker endpoint info (auto-generated)
├── elasticbeanstalk_info.json          # EB environment info (auto-generated)
├── .env                                # Environment variables (not in repo)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- Docker (must be running)
- AWS CLI configured (`aws login`)

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

## 🛠️ Running the Project

### Local Mode (no AWS required)

Train the model and run the app locally:

```bash
python -m src.pipelines.training_pipeline
streamlit run main_local.py
```

---

### AWS Mode — Step by Step

#### Step 1: Configure environment variables

Create a `.env` file in the project root:

```bash
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-...
EB_S3_BUCKET=elasticbeanstalk-eu-north-1-<your-account-id>
```

Find your SageMaker role ARN:
```bash
aws iam list-roles \
  --query "Roles[?contains(RoleName, 'SageMaker')].[RoleName, Arn]" \
  --output table
```

---

#### Step 2: Train + deploy SageMaker endpoint

```bash
python -m src.sagemaker_deployment.orchestrator
```

Trains the model, packages it, builds a Docker image, pushes to ECR, and deploys a SageMaker endpoint. Saves `deployment_info.json`.

#### Step 3: Deploy Streamlit app to Elastic Beanstalk

```bash
python -m src.elastic_beanstalk.deploy_elasticbeanstalk
```

Builds the Streamlit Docker image, pushes to ECR, creates the EB environment with all env vars set. Saves `elasticbeanstalk_info.json` with the app URL.

---

### AWS Mode — One Command (Alternative)

If you want to run all of the above in a single shot:

```bash
python main_orchestrator.py
```

Runs: train -> deploy SageMaker -> deploy Elastic Beanstalk.

---

## 🔐 IAM Permissions Required

The project uses **two** IAM identities. Both must be configured correctly before deployment.

### 1. Your CLI identity (runs deployment scripts)

Must have permissions for: ECR (create repos, push images), S3 (upload bundles), SageMaker (create/delete endpoints), Elastic Beanstalk (create/update environments), IAM `PassRole`, and STS `GetCallerIdentity`.

### 2. EC2 Instance Profile — `testEC2Role` (used by the EB instance at runtime)

This role is attached to the EC2 instance running the Streamlit container. It must have:

| Policy | Purpose |
|---|---|
| `AmazonSageMakerFullAccess` | Invoke SageMaker endpoint cross-region (us-east-1) |
| `AmazonS3FullAccess` | Access EB S3 bucket for artifacts |
| `AmazonEC2ContainerServiceforEC2Role` | Pull Docker image from ECR |
| `CloudWatchLogsFullAccess` | Write logs to CloudWatch log groups |

Attach policies:
```bash
aws iam attach-role-policy \
  --role-name testEC2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name testEC2Role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

aws iam attach-role-policy \
  --role-name testEC2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name testEC2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerServiceforEC2Role
```

Verify what's attached:
```bash
aws iam list-attached-role-policies --role-name testEC2Role
```

### 3. SageMaker Execution Role (used by SageMaker to serve the endpoint)

Needs: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `AmazonEC2ContainerRegistryReadOnly`. This role ARN goes into your `.env` as `SAGEMAKER_ROLE_ARN`.

---

## 📊 Monitoring & Debugging

### CloudWatch Logs (live tail)

```bash
aws logs tail /volatility-predictor/ui --region eu-north-1 --follow
aws logs tail /volatility-predictor/data --region eu-north-1 --follow
aws logs tail /volatility-predictor/inference --region eu-north-1 --follow
```

### Check EB environment health

```bash
aws elasticbeanstalk describe-environments \
  --application-name volatility-predictor-app \
  --environment-name volatility-predictor-app-env \
  --region eu-north-1 \
  --query "Environments[0].{Status:Status,Health:Health,CNAME:CNAME}" \
  --output table
```

### Verify env vars are set on EB

```bash
aws elasticbeanstalk describe-configuration-settings \
  --application-name volatility-predictor-app \
  --environment-name volatility-predictor-app-env \
  --region eu-north-1 \
  --query "ConfigurationSettings[0].OptionSettings[?Namespace=='aws:elasticbeanstalk:application:environment']"
```

Should return all 4 vars: `SAGEMAKER_ENDPOINT_NAME`, `AWS_DEFAULT_REGION`, `CLOUDWATCH_REGION`, `ENABLE_CLOUD_LOGGING`.

### Test SageMaker endpoint directly

```bash
python -m src.sagemaker_deployment.test_endpoint
```

---

## 🔄 Automated Retraining with Airflow

The project includes an Airflow DAG that runs weekly and automatically monitors model performance, downloads fresh data from Kaggle if available, retrains the model if degradation is detected, and redeploys to SageMaker.

### Setup

**1. Create `airflow.env` in the project root:**

```bash
# Airflow Configuration
# Set the home directory to your current project path
export AIRFLOW_HOME=$(pwd)

# Point Airflow to your local DAGs directory
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/src/dags

# Disable macOS fork safety warning (required on Apple Silicon / macOS)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Don't load Airflow example DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=False
```

**2. Initialize the database and launch Airflow:**

```bash
source airflow.env
airflow db migrate
airflow standalone
```

`airflow standalone` starts the webserver, scheduler, and dag-processor in one command. On first run it will print a generated admin password to the terminal — save it.

Open `http://localhost:8080` and log in with username `admin` and the printed password.

### DAG: `volatility_retraining_pipeline`

The DAG runs every week and follows this flow:

```
run_monitoring
      │
check_degradation
      ├── [no degradation] ────────────────────────────────────► end
      └── [degradation]    ──► download_data
                                    │
                               check_new_data
                                    ├── [no new data] ─────────► end
                                    └── [new data]    ──► run_training ──► deploy_to_sagemaker ──► end
```

Each step is an independent Airflow task — if deployment fails, only that task retries without re-running monitoring or training.

You can also trigger a manual run anytime from the Airflow UI by clicking the button next to the DAG.

---

## 🧹 Cleanup

Stop all running AWS resources to avoid charges:

```bash
python cleanup_resources.py
```

Or individually:
```bash
python -m src.elastic_beanstalk.terminate_elasticbeanstalk
python -m src.sagemaker_deployment.terminate_endpoint
```

---

## 📈 Experiment Tracking with MLflow

Every training and retraining run is automatically tracked with MLflow — metrics, hyperparameters, confusion matrices, drift detection analysis, and model artifacts are all logged. To inspect results, launch the MLflow UI from the project root:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser. Under the `volatility_prediction` experiment you'll find all historical runs with their train/val/test F1 scores, accuracy, per-class metrics, and normalized confusion matrix plots. The `volatility_monitoring` experiment tracks all monitoring runs separately, showing performance over time and whether degradation was detected. This makes it easy to compare model versions, spot when performance started to degrade, and decide whether a retrained model is actually better before deploying it.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `eda.ipynb` | Exploratory data analysis — price trends, correlations, statistics |
| `volatility_prediction_model.ipynb` | Volatility regression + classification experiments |
| `binary_classification_model.ipynb` | Experimental up/down price direction classifier |
| `price_prediction_model.ipynb` | Experimental raw price forecasting (time-series / ML / DL) |

---

## ⚙️ Configuration

All settings are in `src/config/config.yaml` — model hyperparameters, API settings, deployment regions, instance types, monitoring thresholds, and UI color maps.