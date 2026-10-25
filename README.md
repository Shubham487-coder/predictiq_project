# PredictIQ — Industrial Predictive Maintenance Platform

> A production-grade AI platform that monitors industrial machines in real time and predicts failures before they happen — built with a full MLOps pipeline, deployed on AWS, and orchestrated end-to-end with Apache Airflow.

**Live Dashboard:** http://51.20.96.40:8501

**Airflow UI:** http://51.20.96.40:8081 (credentials available on request)

---

## The Problem This Solves

In industrial environments, machines fail unexpectedly. When a pump, compressor or motor breaks down without warning, the consequences are severe — unplanned downtime, emergency repair costs, and in some industries, safety risks. Traditional maintenance is done on a fixed schedule: every 3 months, whether the machine needs it or not. This is wasteful and still misses failures that happen between service intervals.

**Predictive maintenance** solves this by continuously monitoring sensor data from machines and using machine learning to answer two questions:

1. **Is this machine going to fail soon, and which component will fail?**
2. **How many hours does it have left before failure?**

PredictIQ answers both questions for 100 machines simultaneously, running on live sensor data, and presents the results in an interactive dashboard that a maintenance team can act on.

---

## What the System Does

PredictIQ ingests telemetry data from 100 industrial machines — sensor readings for voltage, rotation speed, pressure and vibration — sampled every hour over a period of approximately 12 months. That gives a dataset of **868,000 rows**.

From this raw sensor data, the system:

1. **Engineers features** — computes rolling statistics (24-hour and 48-hour moving averages and standard deviations) for each sensor signal on each machine. These rolling features capture trends and degradation patterns that raw point-in-time readings miss.

2. **Classifies which component will fail** — an XGBoost classifier predicts whether a machine will experience a failure in the near future, and if so, which specific component (component 1, 2, 3 or 4). XGBoost was chosen for its speed, interpretability and strong performance on tabular sensor data.

3. **Predicts remaining useful life (RUL)** — a BiLSTM neural network with an attention mechanism predicts the number of hours remaining before failure. The BiLSTM (Bidirectional Long Short-Term Memory) architecture is well suited to time series because it processes the sequence of sensor readings both forwards and backwards, capturing temporal dependencies that simpler models miss. The attention mechanism lets the model focus on the most informative time steps.

4. **Stores results in PostgreSQL** — predictions are persisted to a relational database so the dashboard can query them efficiently and the history of predictions is retained.

5. **Serves predictions via FastAPI** — a REST API runs the XGBoost and BiLSTM models. The Airflow pipeline sends sensor data to the API, gets predictions back, and writes them to PostgreSQL. The dashboard then reads directly from the database to display results.

6. **Displays results in Streamlit** — an interactive dashboard gives maintenance teams a clear view of which machines are at risk, which component is predicted to fail next, and how many hours of useful life remain. Users can set a simulation datetime to see how predictions change at any historical point.

---

## Architecture

The system is made up of 6 Docker containers orchestrated with Docker Compose, running on an AWS EC2 instance:

```
Raw Sensor CSVs (uploaded once to server)
        |
        v
+-----------------------------------------------+
|           Apache Airflow DAG                  |
|           (runs weekly / on demand)           |
|                                               |
|  1. api_health_check                          |
|        |                                      |
|        v                                      |
|  2. load_and_enrich                           |
|     (868k rows, cached after first run)       |
|        |                                      |
|        +------------------+                   |
|        |                  |  (parallel)       |
|        v                  v                   |
|  3. build_classifier_  4. build_rul_input     |
|     input (XGBoost)       (BiLSTM sequences)  |
|        |                  |                   |
|        +------------------+                   |
|                  |                            |
|                  v                            |
|  5. api_calls (fan-out across 100 machines)   |
|     XGBoost + BiLSTM inference via FastAPI    |
|                  |                            |
|                  v                            |
|  6. save_to_postgres                          |
+-----------------------------------------------+
        |
        v
   FastAPI Service  <-->  PostgreSQL
        |
        v
   Streamlit Dashboard  (live on AWS EC2)
```

Every component runs in its own Docker container. The containers communicate over an internal Docker network. The only ports exposed to the internet are the dashboard (8501) and the Airflow web UI (8081).

---

## Machine Learning Details

### Failure Classification — XGBoost

- **Input:** 52 features per machine snapshot (raw sensor readings + rolling statistics across 24h, 48h and 72h windows, error event counts, days since last component maintenance, machine age, model type, and derived interaction features)
- **Output:** Which component will fail next (none, component 1, 2, 3 or 4)
- **Why XGBoost:** Fast inference, handles tabular data extremely well, robust to feature scaling, and produces calibrated probabilities that make it easy to set alert thresholds

### Remaining Useful Life — BiLSTM with Attention

- **Input:** A sequence of the last 48 hourly sensor readings per machine (10 features per timestep: raw sensor values + 6h rolling means + error count)
- **Output:** Predicted hours until next failure (regression)
- **Architecture:** Bidirectional LSTM processes the time series in both directions, capturing both recent trends and longer-term degradation patterns. An attention layer then weights the most informative time steps before the final regression head.
- **Why BiLSTM:** RUL prediction is fundamentally a sequence problem — the trajectory of sensor readings over the past 48 hours matters more than any single reading. BiLSTMs are a standard and well-validated architecture for this class of problem.

---

## Pipeline and Orchestration

The end-to-end pipeline is orchestrated by Apache Airflow. The DAG (Directed Acyclic Graph) defines the sequence of tasks and their dependencies:

- **api_health_check** runs first and blocks the pipeline if the FastAPI service is not healthy. This prevents wasted compute if the inference service is down.
- **load_and_enrich** builds the full feature matrix. On the first run this processes all 868,000 raw rows, which takes a few minutes. On every subsequent run, it loads a cached CSV in seconds — the cache stores the full unfiltered dataset, and the simulation cutoff filter is applied after loading.
- **build_classifier_input** and **build_rul_input** run **in parallel** after the features are ready — both depend on the enriched features but are independent of each other, so Airflow runs them simultaneously to save time. They prepare the input formats required by each model respectively.
- **api_calls** fans out across all 100 machines, sending batch inference requests to FastAPI. Each machine's prediction is independent, so this step is parallelised.
- **save_to_postgres** collects all predictions and writes them to the database in a single batch operation.

---

## CI/CD Pipeline

Every push to the `main` branch triggers a GitHub Actions workflow with two stages:

1. **Run Tests** — installs dependencies and runs a Python syntax check across all critical modules (API, DAG, pipeline tasks). This catches import errors and syntax mistakes before they reach production.

2. **Deploy to EC2** — if tests pass, GitHub Actions SSHes into the EC2 instance, pulls the latest code, and restarts the affected Docker containers. This means a code change goes from a local commit to live on AWS in under 2 minutes, with no manual steps.

The deploy stage is gated on the test stage — broken code never reaches production.

---

## AWS Deployment

The full stack runs on an **AWS EC2 m7i-flex.large** instance (2 vCPU, 8GB RAM) in the eu-north-1 region. This instance type was chosen because it provides enough memory to run all 6 Docker containers simultaneously while remaining cost-effective for a portfolio deployment.

Key infrastructure decisions:
- **Swap space** (4GB) is configured on the instance to handle peak memory usage during the first pipeline run, when the full feature matrix is being built from raw data
- **Security groups** restrict inbound traffic to only the necessary ports (SSH, dashboard, Airflow UI)
- The `.env` file containing credentials is never committed to the repository — it lives only on the server

---

## Tech Stack Summary

| Component | Technology | Why |
|---|---|---|
| Failure classifier | XGBoost | Fast, accurate on tabular sensor data |
| RUL predictor | PyTorch BiLSTM + Attention | Captures temporal dependencies in sensor sequences |
| Pipeline orchestration | Apache Airflow | DAG-based scheduling with task dependencies and retries |
| REST API | FastAPI | Async, high-performance, automatic OpenAPI docs |
| Dashboard | Streamlit | Rapid interactive UI with minimal frontend code |
| Database | PostgreSQL | Reliable relational store for prediction history |
| Containerisation | Docker Compose | Reproducible 6-service stack, one command to run |
| Cloud | AWS EC2 | Real deployment, publicly accessible |
| CI/CD | GitHub Actions | Automated test and deploy on every push |

---

## Project Structure

```
predictiq/
│
├── airflow_tasks/              # Pipeline task logic (called by Airflow)
│   ├── load_and_enrich.py      # Feature engineering + caching
│   ├── build_classifier_input.py
│   ├── build_rul_input.py
│   └── save_to_postgres.py
│
├── api/                        # FastAPI inference service
│   ├── main.py                 # API routes and health check
│   └── models/                 # Pydantic request/response schemas
│
├── dags/
│   └── predictiq_pipeline.py   # Airflow DAG definition
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── ml_models/                  # Model training code
│   ├── feature_engineering.py  # Feature computation from raw sensor data
│   ├── failure_classification/ # XGBoost training
│   └── rul_regression/         # BiLSTM training (multiple experiments)
│
├── docker/                     # Dockerfiles for each service
├── docker-compose.yml          # Full stack orchestration
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions CI/CD pipeline
└── requirements.txt
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Shubham487-coder/predictiq_project.git
cd predictiq_project

# Create .env file (see .env.example for required variables)
cp .env.example .env
# Edit .env with your credentials

# Start all services
docker compose up --build

# Dashboard:  http://localhost:8501
# Airflow UI: http://localhost:8081  (admin / your_password)
# API docs:   http://localhost:8000/docs
```

Once all containers are healthy, trigger the pipeline from the dashboard sidebar or from the Airflow UI. The first run will take a few minutes to build the feature cache; subsequent runs are fast.

---

## Dataset

The project uses the [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) from Kaggle, which contains telemetry, failure records, maintenance logs and machine metadata for 100 machines over approximately 12 months.

The raw CSV files are not included in this repository due to size. Download them from Kaggle and place them in `data/raw/` before running the pipeline.