# ============================================================
# PredictIQ — Airflow Service
# ============================================================
# Built on the official Apache Airflow image with extra providers
# and the project's own packages added on top.
# ============================================================

FROM apache/airflow:2.9.3-python3.11

USER root

# System deps needed for psycopg2 and httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Airflow providers for PostgreSQL + HTTP connections
RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres==5.11.0 \
    apache-airflow-providers-http==4.11.0

# Project runtime deps (only what airflow tasks need, not torch)
# Torch is NOT needed in the Airflow container — tasks call the FastAPI
# service over HTTP. Only psycopg2, httpx, pandas, numpy needed.
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    httpx==0.27.2 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    xgboost==2.1.1 \
    mlflow==2.16.2 \
    scikit-learn==1.5.2

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu

WORKDIR /opt/airflow

# Copy DAGs and task modules
COPY --chown=airflow:root dags/           ./dags/
COPY --chown=airflow:root airflow_tasks/  ./airflow_tasks/
COPY --chown=airflow:root ml_models/      ./ml_models/

# Data directory — shared via volume
RUN mkdir -p /opt/airflow/data/processed /opt/airflow/data/raw

# Entrypoint script
USER root
COPY docker/airflow/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r//' /entrypoint.sh && chmod +x /entrypoint.sh
USER airflow

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$(hostname)" || exit 1

ENTRYPOINT ["/entrypoint.sh"]