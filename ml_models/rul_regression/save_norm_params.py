from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── same constants as train_rul_multihead_v2.py ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
RAW_DIR    = BASE_DIR / "data" / "raw"
MODEL_DIR  = BASE_DIR / "ml_models" / "rul_regression"

SENSOR_COLS  = ["volt", "rotate", "pressure", "vibration"]
FEATURE_COLS = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_6h", "rotate_mean_6h",
    "pressure_mean_6h", "vibration_mean_6h",
    "error_count_6h",
]
SEQUENCE_LENGTH = 48
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END   = pd.Timestamp("2015-08-01")

# ── load raw data (exactly as training script does) ──────────────────────────
print("Loading raw data...")
telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
errors    = pd.read_csv(RAW_DIR / "PdM_errors.csv",    parse_dates=["datetime"])
telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
errors    = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)

# ── engineer 6h features (exactly as training script does) ───────────────────
print("Engineering 6h features...")
enriched = telemetry.copy()
grouped  = enriched.groupby("machineID", group_keys=False)

for sensor in SENSOR_COLS:
    enriched[f"{sensor}_mean_6h"] = grouped[sensor].transform(
        lambda series: series.shift(1).rolling(window=6, min_periods=6).mean()
    )

error_counts = (
    errors.assign(error_count=1)
    .groupby(["machineID", "datetime"])["error_count"]
    .sum()
    .reset_index()
)
enriched = enriched.merge(error_counts, on=["machineID", "datetime"], how="left")
enriched["error_count"] = enriched["error_count"].fillna(0)
enriched["error_count_6h"] = enriched.groupby("machineID")["error_count"].transform(
    lambda series: series.shift(1).rolling(window=6, min_periods=6).sum()
)
enriched = enriched.drop(columns=["error_count"]).dropna().reset_index(drop=True)

# ── build sequences and extract training split ────────────────────────────────
print("Building sequences...")
sequences = []
for _, group in enriched.groupby("machineID"):
    values = group[FEATURE_COLS].values
    for i in range(len(values) - SEQUENCE_LENGTH + 1):
        seq_end_datetime = group["datetime"].iloc[i + SEQUENCE_LENGTH - 1]
        if TRAIN_START <= seq_end_datetime < TRAIN_END:
            sequences.append(values[i : i + SEQUENCE_LENGTH])

X_train = np.array(sequences, dtype=np.float32)
print(f"Training sequences shape: {X_train.shape}")

# ── compute mean and std exactly as split_and_scale does ─────────────────────
mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
std[std == 0] = 1.0

print("\nMean per feature:")
for name, val in zip(FEATURE_COLS, mean):
    print(f"  {name}: {val:.4f}")

print("\nStd per feature:")
for name, val in zip(FEATURE_COLS, std):
    print(f"  {name}: {val:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = MODEL_DIR / "rul_norm_params.npz"
np.savez(out_path, mean=mean, std=std, feature_names=np.array(FEATURE_COLS))
print(f"\nSaved to {out_path}")