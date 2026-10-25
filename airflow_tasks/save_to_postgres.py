"""
Task 6: save_to_postgres
Reads classifier_results.json and rul_results.json, inserts all rows
into predictiq_db with a shared pipeline_run_at timestamp.

Airflow integration:
  - XCom: pulls classifier_results.json path from Task 4 (run_classifier)
  - XCom: pulls rul_results.json path from Task 5 (run_rul)
  - Airflow Connection 'postgres_predictiq' provides DB credentials
    (falls back to environment variables for standalone use)
  - Returns pipeline_run_at ISO string via XCom
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FALLBACK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
_FALLBACK_DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "127.0.0.1"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   "predictiq_db",
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}


def _get_config(context: dict) -> tuple[dict, Path, Path]:
    """Returns (db_config, classifier_path, rul_path)."""
    db_config       = _FALLBACK_DB_CONFIG.copy()
    classifier_path = _FALLBACK_PROCESSED_DIR / "classifier_results.json"
    rul_path        = _FALLBACK_PROCESSED_DIR / "rul_results.json"

    try:
        from airflow.hooks.base import BaseHook

        try:
            conn_obj = BaseHook.get_connection("postgres_predictiq")
            db_config = {
                "host":     conn_obj.host,
                "port":     int(conn_obj.port or 5432),
                "dbname":   conn_obj.schema,
                "user":     conn_obj.login,
                "password": conn_obj.password,
            }
        except Exception:
            pass

        ti = context.get("ti")
        if ti:
            xcom_classifier = ti.xcom_pull(task_ids="api_calls.run_classifier")
            xcom_rul        = ti.xcom_pull(task_ids="api_calls.run_rul")
            if xcom_classifier:
                classifier_path = Path(xcom_classifier)
            if xcom_rul:
                rul_path = Path(xcom_rul)

    except Exception:
        pass

    return db_config, classifier_path, rul_path


def run(**context):
    db_config, classifier_path, rul_path = _get_config(context)
    run_at = datetime.now(timezone.utc)

    print(f"[save_to_postgres] Loading {classifier_path}...")
    with open(classifier_path) as f:
        classifier_results = json.load(f)

    print(f"[save_to_postgres] Loading {rul_path}...")
    with open(rul_path) as f:
        rul_results = json.load(f)

    print(f"[save_to_postgres] {len(classifier_results)} classifier rows, "
          f"{len(rul_results)} RUL rows")

    conn = psycopg2.connect(**db_config)
    try:
        with conn:
            cur = conn.cursor()

            classifier_sql = """
                INSERT INTO classifier_results (
                    machine_id, overall_status, priority,
                    comp1_prob, comp1_alert, comp1_status,
                    comp2_prob, comp2_alert, comp2_status,
                    comp3_prob, comp3_alert, comp3_status,
                    comp4_prob, comp4_alert, comp4_status,
                    pipeline_run_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s
                )
            """
            classifier_rows = [
                (
                    r["machine_id"], r["overall_status"], r["priority"],
                    r["comp1"]["failure_probability"], r["comp1"]["alert"], r["comp1"]["status"],
                    r["comp2"]["failure_probability"], r["comp2"]["alert"], r["comp2"]["status"],
                    r["comp3"]["failure_probability"], r["comp3"]["alert"], r["comp3"]["status"],
                    r["comp4"]["failure_probability"], r["comp4"]["alert"], r["comp4"]["status"],
                    run_at,
                )
                for r in classifier_results
            ]
            cur.executemany(classifier_sql, classifier_rows)
            print(f"[save_to_postgres] Inserted {len(classifier_rows)} classifier rows")

            rul_sql = """
                INSERT INTO rul_results (
                    machine_id, component,
                    predicted_rul_hours, day_bucket, day_label, interpretation,
                    pipeline_run_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            rul_rows = [
                (
                    r["machine_id"], r["component"],
                    r["predicted_rul_hours"], r["day_bucket"],
                    r["day_label"], r["interpretation"],
                    run_at,
                )
                for r in rul_results
            ]
            cur.executemany(rul_sql, rul_rows)
            print(f"[save_to_postgres] Inserted {len(rul_rows)} RUL rows")

    finally:
        conn.close()

    print(f"[save_to_postgres] Done. pipeline_run_at={run_at.isoformat()}")
    return run_at.isoformat()


if __name__ == "__main__":
    run()