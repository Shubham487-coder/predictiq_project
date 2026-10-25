"""
Test script for PredictIQ API.
Run with: python test_api.py
Make sure uvicorn is running in another terminal first.
"""
from __future__ import annotations

import requests
import pandas as pd
import numpy as np
import xgboost as xgb

BASE_URL = "http://localhost:8000/api/v1"

INT_COLS = {"age", "model_model1", "model_model2", "model_model3", "model_model4"}

# ---------------------------------------------------------------------------
# Load real data
# ---------------------------------------------------------------------------

print("Loading features.parquet...")
df = pd.read_parquet("data/processed/features.parquet")
print(f"Loaded {len(df)} rows\n")

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

RUL_FEATURES = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_6h", "rotate_mean_6h", "pressure_mean_6h", "vibration_mean_6h",
    "error_count_6h",
]


def row_to_payload(row, machine_id: int) -> dict:
    payload = {"machine_id": machine_id, "alert_threshold": 0.7}
    for col in CLASSIFIER_FEATURES:
        payload[col] = int(row[col]) if col in INT_COLS else float(row[col])
    return payload


# ---------------------------------------------------------------------------
# Test 1 — Health check
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST 1: Health check")
print("=" * 60)
r = requests.get(f"{BASE_URL}/health")
print(f"Status code: {r.status_code}")
data = r.json()
print(f"API status:     {data['data']['status']}")
print(f"Models loaded:  {data['data']['models_loaded']}")
print(f"Classifiers:    {data['data']['classifier_models']}")
print(f"RUL model:      {data['data']['rul_model']}")
print(f"Uptime:         {data['data']['uptime_seconds']:.1f}s")
print()

# ---------------------------------------------------------------------------
# Test 2 — Pydantic validation (should reject bad data)
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST 2: Pydantic validation — bad data should be rejected")
print("=" * 60)

clean_df = df.dropna(subset=CLASSIFIER_FEATURES)
row = clean_df.iloc[0]

# Bad 1: voltage out of range
payload = row_to_payload(row, machine_id=1)
payload["volt"] = 9999.0
r = requests.post(f"{BASE_URL}/classify", json=payload)
print(f"Out-of-range voltage       → status {r.status_code} (expect 422)")

# Bad 2: all-zero model flags
payload = row_to_payload(row, machine_id=1)
payload["model_model1"] = 0
payload["model_model2"] = 0
payload["model_model3"] = 0
payload["model_model4"] = 0
r = requests.post(f"{BASE_URL}/classify", json=payload)
print(f"All-zero model flags       → status {r.status_code} (expect 422)")

# Bad 3: two model flags set
payload = row_to_payload(row, machine_id=1)
payload["model_model1"] = 1
payload["model_model2"] = 1
payload["model_model3"] = 0
payload["model_model4"] = 0
r = requests.post(f"{BASE_URL}/classify", json=payload)
print(f"Two model flags set        → status {r.status_code} (expect 422)")

# Bad 4: fractional error count
payload = row_to_payload(row, machine_id=1)
payload["error1_count_24h"] = 0.5
r = requests.post(f"{BASE_URL}/classify", json=payload)
print(f"Fractional error count     → status {r.status_code} (expect 422)")
print()

# ---------------------------------------------------------------------------
# Test 3 — Classifier with real data
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST 3: Classifier with real data")
print("=" * 60)

healthy_row = clean_df[
    (clean_df["comp1_fail_24h"] == 0) &
    (clean_df["comp2_fail_24h"] == 0) &
    (clean_df["comp3_fail_24h"] == 0) &
    (clean_df["comp4_fail_24h"] == 0)
].iloc[100]

failure_row = clean_df[
    (clean_df["comp1_fail_24h"] == 1) |
    (clean_df["comp2_fail_24h"] == 1) |
    (clean_df["comp3_fail_24h"] == 1) |
    (clean_df["comp4_fail_24h"] == 1)
].iloc[0]

for label, row in [("HEALTHY machine", healthy_row), ("NEAR-FAILURE machine", failure_row)]:
    payload = row_to_payload(row, machine_id=int(row["machineID"]))
    r = requests.post(f"{BASE_URL}/classify", json=payload)
    data = r.json()["data"]
    print(f"{label} (machineID={int(row['machineID'])}):")
    print(f"  Overall: {data['overall_status']} (priority {data['priority']})")
    for comp in ["comp1", "comp2", "comp3", "comp4"]:
        p = data[comp]
        print(f"  {comp}: prob={p['failure_probability']:.4f} alert={p['alert']} status={p['status']}")
    print()

# ---------------------------------------------------------------------------
# Test 4 — Verify classifier matches direct XGBoost prediction
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST 4: API vs direct XGBoost — must match exactly")
print("=" * 60)

xgb_models = {
    "comp1": xgb.Booster(),
    "comp2": xgb.Booster(),
    "comp3": xgb.Booster(),
    "comp4": xgb.Booster(),
}
xgb_models["comp1"].load_model("ml_models/failure_classification/xgb_comp1_24h.json")
xgb_models["comp2"].load_model("ml_models/failure_classification/xgb_comp2_24h.json")
xgb_models["comp3"].load_model("ml_models/failure_classification/xgb_comp3_24h.json")
xgb_models["comp4"].load_model("ml_models/failure_classification/xgb_comp4_24h.json")

row = clean_df.iloc[500]
features = [int(row[col]) if col in INT_COLS else float(row[col]) for col in CLASSIFIER_FEATURES]

# Use named DataFrame — same as the API does internally
df_pred = pd.DataFrame([features], columns=CLASSIFIER_FEATURES)
dmatrix = xgb.DMatrix(df_pred)

print("Direct XGBoost predictions:")
direct = {}
for comp, model in xgb_models.items():
    prob = float(model.predict(dmatrix)[0])
    direct[comp] = prob
    print(f"  {comp}: {prob:.6f}")

payload = row_to_payload(row, machine_id=int(row["machineID"]))
r = requests.post(f"{BASE_URL}/classify", json=payload)
api_data = r.json()["data"]

print("\nAPI predictions:")
for comp in ["comp1", "comp2", "comp3", "comp4"]:
    prob = api_data[comp]["failure_probability"]
    print(f"  {comp}: {prob:.6f}")

print("\nMatch check:")
all_match = True
for comp in ["comp1", "comp2", "comp3", "comp4"]:
    api_prob = api_data[comp]["failure_probability"]
    direct_prob = round(direct[comp], 4)
    match = abs(api_prob - direct_prob) < 1e-3
    if not match:
        all_match = False
    print(f"  {comp}: {'✓ MATCH' if match else '✗ MISMATCH'} (api={api_prob:.4f} direct={direct_prob:.4f})")

print()
if all_match:
    print("✓ All tests passed. API predictions match XGBoost exactly.")
else:
    print("✗ Some predictions do not match. Check the feature order.")