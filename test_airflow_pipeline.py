"""
test_airflow_pipeline.py — PredictIQ Airflow pipeline integration test.

For 10 as_of timestamps, runs the pipeline two completely independent ways
and compares results:

  Path A (Direct): Uses raw source code ONLY — ml_models/feature_engineering.py
                   for features, builds inputs manually, calls FastAPI directly
                   with httpx. Zero imports from airflow_tasks/.

  Path B (DAG):    Runs `airflow dags test predictiq_pipeline` with the same
                   as_of conf, reads results from PostgreSQL (rows inserted by
                   that exact run, identified by pipeline_run_at timestamp).

If classifier probabilities match to 4dp and RUL hours match to 2dp
for all 10 timestamps → Airflow pipeline is proven correct end-to-end.

Run with:
    source ~/predictiq_venv/bin/activate
    python test_airflow_pipeline.py

Make sure FastAPI (uvicorn) is running before starting.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Raw source imports (NO airflow_tasks imports) ─────────────────────────────
from ml_models.feature_engineering import build_features
from ml_models.rul_regression.train_rul_multihead_v2 import (
    enrich_telemetry,
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    COMPONENTS,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
API_BASE = "http://localhost:8000/api/v1"
BATCH_SIZE = 100
TIMEOUT = 120

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "predictiq_db",
    "user": "postgres",
    "password": "Goodluck@1234",
}

AS_OF_LIST = [
    "2015-02-01 06:00:00",
    "2015-03-15 12:00:00",
    "2015-04-10 18:00:00",
    "2015-05-20 00:00:00",
    "2015-06-05 08:00:00",
    "2015-07-01 14:00:00",
    "2015-08-10 20:00:00",
    "2015-09-22 10:00:00",
    "2015-10-30 16:00:00",
    "2015-11-15 02:00:00",
]

CLASSIFIER_PROB_TOL = 1e-4
RUL_HOURS_TOL = 0.01

CLASSIFIER_FEATURES = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_24h", "volt_std_24h", "rotate_mean_24h", "rotate_std_24h",
    "pressure_mean_24h", "pressure_std_24h", "vibration_mean_24h", "vibration_std_24h",
    "volt_mean_48h", "volt_std_48h", "rotate_mean_48h", "rotate_std_48h",
    "pressure_mean_48h", "pressure_std_48h", "vibration_mean_48h", "vibration_std_48h",
    "volt_mean_72h", "volt_std_72h", "rotate_mean_72h", "rotate_std_72h",
    "pressure_mean_72h", "pressure_std_72h", "vibration_mean_72h", "vibration_std_72h",
    "error1_count_24h", "error2_count_24h", "error3_count_24h", "error4_count_24h", "error5_count_24h",
    "error1_count_48h", "error2_count_48h", "error3_count_48h", "error4_count_48h", "error5_count_48h",
    "error1_count_72h", "error2_count_72h", "error3_count_72h", "error4_count_72h", "error5_count_72h",
    "days_since_comp1_maint", "days_since_comp2_maint", "days_since_comp3_maint", "days_since_comp4_maint",
    "age",
    "model_model1", "model_model2", "model_model3", "model_model4",
    "volt_rotate_ratio_24h", "vibration_cv_24h", "error_count_total_72h", "hours_since_last_error",
]

CLIP_BOUNDS = {
    "days_since_comp1_maint": (0.0, 492.0),
    "days_since_comp2_maint": (0.0, 349.0),
    "days_since_comp3_maint": (0.0, 371.0),
    "days_since_comp4_maint": (0.0, 395.0),
    "volt_rotate_ratio_24h": (0.33, 0.64),
    "vibration_cv_24h": (0.04, 0.26),
    "hours_since_last_error": (1.0, 9999.0),
    "error_count_total_72h": (0.0, 5.0),
}

INT_FIELDS = {"age", "model_model1", "model_model2", "model_model3", "model_model4"}


# ── Path A helpers ────────────────────────────────────────────────────────────

def _build_classifier_records(features_df: pd.DataFrame) -> list[dict]:
    latest = features_df.sort_values("datetime").groupby("machineID").last().reset_index()
    records = []
    for _, row in latest.iterrows():
        record = {"machine_id": int(row["machineID"])}
        for feat in CLASSIFIER_FEATURES:
            val = float(row[feat]) if not pd.isna(row[feat]) else 0.0
            if feat in CLIP_BOUNDS:
                lo, hi = CLIP_BOUNDS[feat]
                val = float(np.clip(val, lo, hi))
            record[feat] = int(val) if feat in INT_FIELDS else val
        records.append(record)
    return records


def _build_rul_records(as_of: str) -> list[dict]:
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    enriched = enrich_telemetry(telemetry, errors)
    cutoff = pd.Timestamp(as_of)
    enriched = enriched[enriched["datetime"] <= cutoff].reset_index(drop=True)

    records = []
    for machine_id in enriched["machineID"].unique():
        machine_data = enriched[enriched["machineID"] == machine_id].tail(SEQUENCE_LENGTH)
        if len(machine_data) < SEQUENCE_LENGTH:
            continue
        sequence = machine_data[FEATURE_COLS].values.tolist()
        for comp in COMPONENTS:
            records.append({
                "machine_id": int(machine_id),
                "component": comp,
                "sensor_sequence": sequence,
            })
    return records


def _call_classifier_api(records: list[dict]) -> list[dict]:
    all_results = []
    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(math.ceil(len(records) / BATCH_SIZE)):
            chunk = records[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            resp = client.post(f"{API_BASE}/classify/batch", json={"requests": chunk})
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Classifier API error: {data.get('error')}")
            all_results.extend(data["data"])
    return all_results


def _call_rul_api(records: list[dict]) -> list[dict]:
    all_results = []
    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(math.ceil(len(records) / BATCH_SIZE)):
            chunk = records[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            resp = client.post(f"{API_BASE}/rul/batch", json={"requests": chunk})
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"RUL API error: {data.get('error')}")
            all_results.extend(data["data"])
    return all_results


def run_direct(as_of: str) -> tuple[list[dict], list[dict]]:
    """Path A: pure raw ml_models code + direct httpx calls. No airflow_tasks."""
    features_df, _, _, _ = build_features()
    cutoff = pd.Timestamp(as_of)
    features_df = features_df[features_df["datetime"] <= cutoff].reset_index(drop=True)

    classifier_records = _build_classifier_records(features_df)
    rul_records = _build_rul_records(as_of)

    return _call_classifier_api(classifier_records), _call_rul_api(rul_records)


# ── Path B: Airflow DAG → PostgreSQL ─────────────────────────────────────────

def run_dag(as_of: str, execution_date: str) -> tuple[list[dict], list[dict]]:
    before = datetime.now(timezone.utc)

    result = subprocess.run(
        ["airflow", "dags", "test", "predictiq_pipeline", execution_date,
         "--conf", json.dumps({"as_of": as_of})],
        capture_output=True, text=True,
    )
    after = datetime.now(timezone.utc)

    if result.returncode != 0:
        print(f"  [DAG STDERR]\n{result.stderr[-3000:]}")
        raise RuntimeError(f"airflow dags test failed for as_of={as_of}")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT machine_id, overall_status, priority,
                   comp1_prob, comp1_alert, comp1_status,
                   comp2_prob, comp2_alert, comp2_status,
                   comp3_prob, comp3_alert, comp3_status,
                   comp4_prob, comp4_alert, comp4_status
            FROM classifier_results
            WHERE pipeline_run_at > %s AND pipeline_run_at <= %s
            ORDER BY machine_id
        """, (before, after))
        classifier_rows = cur.fetchall()

        cur.execute("""
            SELECT machine_id, component, predicted_rul_hours,
                   day_bucket, day_label, interpretation
            FROM rul_results
            WHERE pipeline_run_at > %s AND pipeline_run_at <= %s
            ORDER BY machine_id, component
        """, (before, after))
        rul_rows = cur.fetchall()
    finally:
        conn.close()

    if not classifier_rows:
        raise RuntimeError(f"No classifier rows in DB for as_of={as_of}.")
    if not rul_rows:
        raise RuntimeError(f"No RUL rows in DB for as_of={as_of}.")

    classifier = [
        {
            "machine_id": r[0], "overall_status": r[1], "priority": r[2],
            "comp1": {"failure_probability": r[3], "alert": r[4], "status": r[5]},
            "comp2": {"failure_probability": r[6], "alert": r[7], "status": r[8]},
            "comp3": {"failure_probability": r[9], "alert": r[10], "status": r[11]},
            "comp4": {"failure_probability": r[12], "alert": r[13], "status": r[14]},
        }
        for r in classifier_rows
    ]
    rul = [
        {
            "machine_id": r[0], "component": r[1],
            "predicted_rul_hours": r[2], "day_bucket": r[3],
            "day_label": r[4], "interpretation": r[5],
        }
        for r in rul_rows
    ]
    return classifier, rul


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_classifier(direct: list[dict], dag: list[dict]) -> list[str]:
    mismatches = []
    d_by_id = {r["machine_id"]: r for r in direct}
    g_by_id = {r["machine_id"]: r for r in dag}
    if set(d_by_id) != set(g_by_id):
        mismatches.append(f"machine_id sets differ")
        return mismatches
    for mid in sorted(d_by_id):
        d, g = d_by_id[mid], g_by_id[mid]
        if d["overall_status"] != g["overall_status"]:
            mismatches.append(f"  machine {mid}: overall_status {d['overall_status']} vs {g['overall_status']}")
        if d["priority"] != g["priority"]:
            mismatches.append(f"  machine {mid}: priority {d['priority']} vs {g['priority']}")
        for comp in ["comp1", "comp2", "comp3", "comp4"]:
            dp, gp = d[comp]["failure_probability"], g[comp]["failure_probability"]
            if abs(dp - gp) > CLASSIFIER_PROB_TOL:
                mismatches.append(f"  machine {mid} {comp}: prob {dp:.6f} vs {gp:.6f} (diff={abs(dp - gp):.2e})")
            if d[comp]["alert"] != g[comp]["alert"]:
                mismatches.append(f"  machine {mid} {comp}: alert {d[comp]['alert']} vs {g[comp]['alert']}")
            if d[comp]["status"] != g[comp]["status"]:
                mismatches.append(f"  machine {mid} {comp}: status {d[comp]['status']} vs {g[comp]['status']}")
    return mismatches


def compare_rul(direct: list[dict], dag: list[dict]) -> list[str]:
    mismatches = []
    key = lambda r: (r["machine_id"], r["component"])
    d_by_key = {key(r): r for r in direct}
    g_by_key = {key(r): r for r in dag}
    if set(d_by_key) != set(g_by_key):
        mismatches.append("(machine_id, component) sets differ")
        return mismatches
    for k in sorted(d_by_key):
        d, g = d_by_key[k], g_by_key[k]
        if abs(d["predicted_rul_hours"] - g["predicted_rul_hours"]) > RUL_HOURS_TOL:
            mismatches.append(
                f"  machine {k[0]} {k[1]}: rul_hours {d['predicted_rul_hours']:.4f} vs {g['predicted_rul_hours']:.4f}")
        if d["day_bucket"] != g["day_bucket"]:
            mismatches.append(f"  machine {k[0]} {k[1]}: day_bucket {d['day_bucket']} vs {g['day_bucket']}")
        if d["interpretation"] != g["interpretation"]:
            mismatches.append(f"  machine {k[0]} {k[1]}: interpretation {d['interpretation']} vs {g['interpretation']}")
    return mismatches


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("PredictIQ — Airflow pipeline integration test")
    print("Path A: raw ml_models code + direct httpx API calls (no airflow_tasks)")
    print("Path B: airflow dags test → PostgreSQL")
    print(f"Testing {len(AS_OF_LIST)} as_of timestamps\n")

    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        print(f"FastAPI healthy — models_loaded={r.json()['data']['models_loaded']}\n")
    except Exception as e:
        print(f"✗ FastAPI not reachable: {e}\n  Start: uvicorn api.main:app --reload")
        sys.exit(1)

    all_passed = True
    results_summary = []

    for i, as_of in enumerate(AS_OF_LIST):
        exec_date = f"2026-10-{22 + i:02d}"
        print("=" * 70)
        print(f"[{i + 1:02d}/10] as_of = {as_of}   (exec_date={exec_date})")
        print("=" * 70)

        print("  [A] Running via raw ml_models code + httpx...")
        try:
            dc, dr = run_direct(as_of)
            print(f"      {len(dc)} classifier results, {len(dr)} RUL results")
        except Exception as e:
            print(f"  ✗ Direct path failed: {e}")
            all_passed = False
            results_summary.append((as_of, "DIRECT_FAIL", str(e)))
            continue

        print("  [B] Running via Airflow DAG → PostgreSQL...")
        try:
            gc, gr = run_dag(as_of, exec_date)
            print(f"      {len(gc)} classifier results, {len(gr)} RUL results")
        except Exception as e:
            print(f"  ✗ DAG path failed: {e}")
            all_passed = False
            results_summary.append((as_of, "DAG_FAIL", str(e)))
            continue

        c_miss = compare_classifier(dc, gc)
        r_miss = compare_rul(dr, gr)

        if not c_miss and not r_miss:
            print(f"  ✓ MATCH — {len(dc)} classifier + {len(dr)} RUL results identical")
            results_summary.append((as_of, "PASS", ""))
        else:
            all_passed = False
            total = len(c_miss) + len(r_miss)
            print(f"  ✗ MISMATCH — {total} difference(s):")
            for m in (c_miss + r_miss)[:15]:
                print(f"    {m}")
            results_summary.append((as_of, "MISMATCH", f"{total} differences"))
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for as_of, status, detail in results_summary:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {as_of}  [{status}]" + (f" — {detail}" if detail else ""))

    print()
    if all_passed:
        print("✓ All 10 timestamps passed. Airflow pipeline matches raw code exactly.")
    else:
        passed = sum(1 for _, s, _ in results_summary if s == "PASS")
        print(f"✗ {passed}/10 passed. Check mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
