"""
PredictIQ Airflow DAG — predictiq_pipeline

Runs hourly. Can be triggered manually with a simulation timestamp:
    airflow dags trigger predictiq_pipeline --conf '{"as_of": "2015-06-01 12:00:00"}'

Task topology (t2 and t3 run in parallel after t1):

    api_health_check
         └── load_and_enrich (t1)
                  ├── build_classifier_input (t2) ──→ [api_calls] run_classifier (t4) ──→ save_to_postgres (t6)
                  └── build_rul_input        (t3) ──→ [api_calls] run_rul         (t5) ──→ save_to_postgres (t6)

Required one-time setup:
    airflow connections add 'postgres_predictiq' \
        --conn-type postgres --conn-host 127.0.0.1 --conn-port 5432 \
        --conn-schema predictiq_db --conn-login postgres --conn-password Goodluck@1234

    airflow connections add 'fastapi_predictiq' \
        --conn-type http --conn-host localhost --conn-port 8000 --conn-schema http

    airflow variables set PREDICTIQ_BATCH_SIZE 100
    airflow variables set PREDICTIQ_PROCESSED_DIR '/mnt/c/Users/itMember/Documents/predictiq/data/processed'
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airflow_tasks.load_and_enrich        import run as load_and_enrich
from airflow_tasks.build_classifier_input import run as build_classifier_input
from airflow_tasks.build_rul_input        import run as build_rul_input
from airflow_tasks.run_classifier         import run as run_classifier
from airflow_tasks.run_rul                import run as run_rul
from airflow_tasks.save_to_postgres       import run as save_to_postgres

log = logging.getLogger(__name__)


def _on_failure_callback(context: dict) -> None:
    """
    Called automatically by Airflow when any task fails.
    Logs structured error details — swap log.error for Slack/email/PagerDuty
    hook when moving to production alerting.
    """
    ti        = context["task_instance"]
    exception = context.get("exception")
    log.error(
        "Task FAILED | dag=%s | task=%s | run_id=%s | try=%s | error=%s",
        ti.dag_id, ti.task_id, ti.run_id, ti.try_number, exception,
    )


default_args = {
    "owner":                     "predictiq",
    "retries":                   2,
    "retry_delay":               timedelta(minutes=1),
    "retry_exponential_backoff": True,   # 1m → 2m → 4m
    "email_on_failure":          False,
    "on_failure_callback":       _on_failure_callback,
}

with DAG(
    dag_id="predictiq_pipeline",
    description="Hourly predictive maintenance — classify + RUL for 100 machines",
    schedule="@weekly",
    start_date=datetime(2026, 10, 22),
    catchup=False,
    default_args=default_args,
    tags=["predictiq", "maintenance"],
    doc_md="""
## PredictIQ Pipeline

Runs every hour. Trigger manually with a simulation timestamp:

```bash
airflow dags trigger predictiq_pipeline --conf '{"as_of": "2015-06-01 12:00:00"}'
```

### Task flow

### Connections required
- `fastapi_predictiq`  — HTTP connection (localhost:8000)
- `postgres_predictiq` — PostgreSQL connection (predictiq_db)

### Variables
- `PREDICTIQ_BATCH_SIZE`    — API batch size (default: 100)
- `PREDICTIQ_PROCESSED_DIR` — Override for data/processed directory path

### XCom flow
- `load_and_enrich`        → features_latest.csv path
- `build_classifier_input` → classifier_input.json path
- `build_rul_input`        → rul_input.json path
- `run_classifier`         → classifier_results.json path
- `run_rul`                → rul_results.json path
- `save_to_postgres`       → pipeline_run_at ISO timestamp
""",
) as dag:

    # ── Preflight: verify FastAPI is up and models are loaded ─────────────────
    api_health = HttpSensor(
        task_id="api_health_check",
        http_conn_id="fastapi_predictiq",
        endpoint="/api/v1/health",
        response_check=lambda resp: resp.json().get("data", {}).get("models_loaded") is True,
        poke_interval=15,
        timeout=120,
        mode="poke",
        doc_md="Waits until FastAPI is up and all ML models are loaded before starting.",
    )

    # ── Task 1: Load raw CSVs, build features, filter to as_of ───────────────
    t1 = PythonOperator(
        task_id="load_and_enrich",
        python_callable=load_and_enrich,
        provide_context=True,
        doc_md="Builds 70+ features from raw CSVs and filters to the as_of simulation cutoff.",
    )

    # ── Tasks 2 & 3: Build API inputs — run IN PARALLEL after t1 ─────────────
    t2 = PythonOperator(
        task_id="build_classifier_input",
        python_callable=build_classifier_input,
        provide_context=True,
        doc_md="Latest row per machine → 70-feature ClassifierRequest JSON. Pulls T1 path via XCom.",
    )

    t3 = PythonOperator(
        task_id="build_rul_input",
        python_callable=build_rul_input,
        provide_context=True,
        doc_md="Last 48h sequence per machine×component → RUL input JSON. Pulls T1 dir via XCom.",
    )

    # ── Tasks 4 & 5: FastAPI inference — grouped in Airflow UI ───────────────
    with TaskGroup(
        group_id="api_calls",
        tooltip="FastAPI inference — classifier and RUL run independently",
    ) as api_group:

        t4 = PythonOperator(
            task_id="run_classifier",
            python_callable=run_classifier,
            provide_context=True,
            doc_md="POST /classify/batch → classifier_results.json. Uses 'fastapi_predictiq' connection.",
        )

        t5 = PythonOperator(
            task_id="run_rul",
            python_callable=run_rul,
            provide_context=True,
            doc_md="POST /rul/batch in 4 batches → rul_results.json. Uses 'fastapi_predictiq' connection.",
        )

    # ── Task 6: Persist to PostgreSQL ────────────────────────────────────────
    t6 = PythonOperator(
        task_id="save_to_postgres",
        python_callable=save_to_postgres,
        provide_context=True,
        doc_md="Inserts classifier + RUL results into predictiq_db. Uses 'postgres_predictiq' connection.",
    )

    # ── Dependency graph ──────────────────────────────────────────────────────
    #
    #  api_health → t1 → [t2, t3] (parallel) → api_group → t6
    #
    api_health >> t1 >> [t2, t3] >> api_group >> t6
