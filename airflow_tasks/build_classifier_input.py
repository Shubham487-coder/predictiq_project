"""
Task 2: build_classifier_input
Reads features_latest.csv, takes the most recent row per machine,
builds the 70-feature dict matching ClassifierRequest,
saves to data/processed/classifier_input.json

Airflow integration:
  - XCom: pulls features_latest.csv path from Task 1 (load_and_enrich)
  - PREDICTIQ_PROCESSED_DIR Variable overrides the output directory
  - Returns classifier_input.json path via XCom
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_FALLBACK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Exact order from api/endpoints/classifier.py FEATURE_NAMES
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

# Clip to valid API bounds to avoid validation errors
CLIP_BOUNDS = {
    "days_since_comp1_maint": (0.0, 492.0),
    "days_since_comp2_maint": (0.0, 349.0),
    "days_since_comp3_maint": (0.0, 371.0),
    "days_since_comp4_maint": (0.0, 395.0),
    "volt_rotate_ratio_24h":  (0.33, 0.64),
    "vibration_cv_24h":       (0.04, 0.26),
    "hours_since_last_error": (1.0, 9999.0),
    "error_count_total_72h":  (0.0, 5.0),
}

INT_FIELDS = {"age", "model_model1", "model_model2", "model_model3", "model_model4"}


def _get_paths(context: dict) -> tuple[Path, Path]:
    """Returns (features_csv_path, output_json_path)."""
    try:
        from airflow.models import Variable

        ti = context.get("ti")
        if ti:
            features_path = Path(ti.xcom_pull(task_ids="load_and_enrich"))
            out_path = features_path.parent / "classifier_input.json"
            return features_path, out_path

        p = Variable.get("PREDICTIQ_PROCESSED_DIR", default_var=None)
        if p:
            d = Path(p)
            return d / "features_latest.csv", d / "classifier_input.json"
    except Exception:
        pass

    d = _FALLBACK_PROCESSED_DIR
    return d / "features_latest.csv", d / "classifier_input.json"


def run(**context):
    features_path, out_path = _get_paths(context)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_classifier_input] Reading {features_path}...")
    df = pd.read_csv(features_path, parse_dates=["datetime"])

    # Most recent row per machine
    latest = df.sort_values("datetime").groupby("machineID").last().reset_index()
    print(f"[build_classifier_input] Building inputs for {len(latest)} machines...")

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

    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"[build_classifier_input] Saved {len(records)} records to {out_path}")
    return str(out_path)   # ← XCom: Task 4 pulls this path


if __name__ == "__main__":
    run()