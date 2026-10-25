#!/usr/bin/env bash
# ============================================================
# PredictIQ — Airflow Entrypoint
# ============================================================
# Handles both scheduler and webserver roles based on $AIRFLOW_ROLE.
# On first start (scheduler): initialises DB, creates admin user,
# registers Connections and Variables, then starts the scheduler.
# ============================================================

set -euo pipefail

ROLE="${AIRFLOW_ROLE:-scheduler}"

echo "[entrypoint] PredictIQ Airflow starting as: ${ROLE}"

# ── Wait for PostgreSQL ────────────────────────────────────────
wait_for_postgres() {
    echo "[entrypoint] Waiting for PostgreSQL..."
    local retries=30
    until python3 -c "
import psycopg2, sys
try:
    psycopg2.connect(
        host='${POSTGRES_HOST}',
        port=${POSTGRES_PORT},
        dbname='airflow_db',
        user='${POSTGRES_USER}',
        password='${POSTGRES_PASSWORD}'
    ).close()
    sys.exit(0)
except Exception as e:
    print(f'  not ready: {e}')
    sys.exit(1)
" 2>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -le 0 ]; then
            echo "[entrypoint] ERROR: PostgreSQL not available after 30 tries. Exiting."
            exit 1
        fi
        echo "[entrypoint] PostgreSQL not ready, retrying in 5s... (${retries} left)"
        sleep 5
    done
    echo "[entrypoint] PostgreSQL is ready."
}

# ── One-time Airflow initialisation (only from scheduler) ──────
init_airflow() {
    echo "[entrypoint] Initialising Airflow metadata database..."
    airflow db migrate

    echo "[entrypoint] Creating admin user (idempotent)..."
    airflow users create \
        --username admin \
        --password admin \
        --firstname PredictIQ \
        --lastname Admin \
        --role Admin \
        --email admin@predictiq.local \
        2>/dev/null || echo "[entrypoint] Admin user already exists, skipping."

    echo "[entrypoint] Registering Airflow Connections..."

    # FastAPI connection — points to the API container
    airflow connections delete fastapi_predictiq 2>/dev/null || true
    airflow connections add fastapi_predictiq \
        --conn-type http \
        --conn-host api \
        --conn-port 8000 \
        --conn-schema http

    # PostgreSQL connection — points to the postgres container
    airflow connections delete postgres_predictiq 2>/dev/null || true
    airflow connections add postgres_predictiq \
        --conn-type postgres \
        --conn-host "${POSTGRES_HOST}" \
        --conn-port "${POSTGRES_PORT}" \
        --conn-schema predictiq_db \
        --conn-login "${POSTGRES_USER}" \
        --conn-password "${POSTGRES_PASSWORD}"

    echo "[entrypoint] Registering Airflow Variables..."
    airflow variables set PREDICTIQ_BATCH_SIZE 100
    airflow variables set PREDICTIQ_PROCESSED_DIR /opt/airflow/data/processed

    echo "[entrypoint] Airflow init complete."
}

# ── Wait for FastAPI to be healthy ─────────────────────────────
wait_for_api() {
    echo "[entrypoint] Waiting for FastAPI inference service..."
    local retries=40
    until curl -sf "http://api:8000/api/v1/health" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data',{}).get('models_loaded') else 1)" \
        2>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -le 0 ]; then
            echo "[entrypoint] WARNING: FastAPI not ready after 40 tries. DAG health sensor will handle this."
            break
        fi
        echo "[entrypoint] FastAPI not ready, retrying in 10s... (${retries} left)"
        sleep 10
    done
    echo "[entrypoint] FastAPI is ready."
}

# ── Main ───────────────────────────────────────────────────────
wait_for_postgres

if [ "${ROLE}" = "scheduler" ]; then
    init_airflow
    wait_for_api
    echo "[entrypoint] Starting Airflow Scheduler..."
    exec airflow scheduler

elif [ "${ROLE}" = "webserver" ]; then
    # Webserver needs DB to be ready, init done by scheduler
    echo "[entrypoint] Waiting for DB init (scheduler does this)..."
    sleep 20
    echo "[entrypoint] Starting Airflow Webserver..."
    exec airflow webserver --port 8080

else
    echo "[entrypoint] ERROR: Unknown AIRFLOW_ROLE='${ROLE}'. Use 'scheduler' or 'webserver'."
    exit 1
fi