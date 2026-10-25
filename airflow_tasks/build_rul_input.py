"""
Task 3: build_rul_input
Reuses enrich_telemetry() from train_rul_multihead_v2.py — the exact same
6h feature engineering used during training — then extracts the last 48
rows per machine as the RUL input sequence.
Saves to data/processed/rul_input.json — one entry per (machine, component).

Airflow integration:
  - XCom: pulls features_latest.csv path from Task 1 (load_and_enrich)
          NOTE: build_rul_input re-reads raw CSVs via enrich_telemetry because
          the RUL model needs 6h rolling features, not the 24/48/72h classifier
          features. The XCom path from T1 is used only to locate the processed dir.
  - PREDICTIQ_PROCESSED_DIR Variable overrides the output directory
  - Returns rul_input.json path via XCom
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.rul_regression.train_rul_multihead_v2 import (
    enrich_telemetry,
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    COMPONENTS,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
_FALLBACK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _get_output_path(context: dict) -> Path:
    try:
        from airflow.models import Variable

        ti = context.get("ti")
        if ti:
            features_path = ti.xcom_pull(task_ids="load_and_enrich")
            if features_path:
                return Path(features_path).parent / "rul_input.json"

        p = Variable.get("PREDICTIQ_PROCESSED_DIR", default_var=None)
        if p:
            return Path(p) / "rul_input.json"
    except Exception:
        pass

    return _FALLBACK_PROCESSED_DIR / "rul_input.json"


def run(**context):
    out_path = _get_output_path(context)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Read as_of from Airflow dag_run conf ─────────────────────────────────
    as_of = None
    if context.get("dag_run"):
        as_of = context["dag_run"].conf.get("as_of")

    print("[build_rul_input] Loading raw telemetry and errors...")
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    errors    = pd.read_csv(RAW_DIR / "PdM_errors.csv",    parse_dates=["datetime"])

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors    = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    print("[build_rul_input] Engineering 6h features (reusing training code)...")
    enriched = enrich_telemetry(telemetry, errors)

    # ── Apply as_of cutoff ────────────────────────────────────────────────────
    if as_of:
        cutoff = pd.Timestamp(as_of)
        before = len(enriched)
        enriched = enriched[enriched["datetime"] <= cutoff].reset_index(drop=True)
        print(f"[build_rul_input] Filtered {before} → {len(enriched)} rows (cutoff: {cutoff})")

    print(f"[build_rul_input] Extracting last {SEQUENCE_LENGTH} rows per machine...")
    records = []
    for machine_id in enriched["machineID"].unique():
        machine_data = enriched[enriched["machineID"] == machine_id].tail(SEQUENCE_LENGTH)

        if len(machine_data) < SEQUENCE_LENGTH:
            print(f"  [WARN] machine {machine_id} only has {len(machine_data)} rows — skipping")
            continue

        sequence = machine_data[FEATURE_COLS].values.tolist()

        for comp in COMPONENTS:
            records.append({
                "machine_id": int(machine_id),
                "component":  comp,
                "sequence":   sequence,
            })

    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"[build_rul_input] Saved {len(records)} sequences to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()