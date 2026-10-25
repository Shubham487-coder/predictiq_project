"""
save_norm_params_fixed.py

Recomputes RUL normalization parameters EXACTLY as train_rul_multihead_v2.py
does in split_and_scale() — mean/std from training RUL sequences only
(not all telemetry rows).

Run once after training:
    python ml_models/rul_regression/save_norm_params_fixed.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml_models" / "rul_regression"

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
FEATURE_COLS = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_6h", "rotate_mean_6h",
    "pressure_mean_6h", "vibration_mean_6h",
    "error_count_6h",
]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
SEQUENCE_LENGTH = 48
MAX_RUL_HOURS = 100
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-08-01")

print("Loading raw data...")
telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
maint = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])

for df in [telemetry, failures, maint, errors]:
    df.sort_values(["machineID", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

# ── enrich_telemetry — identical to training ─────────────────────────────────
print("Enriching telemetry...")
enriched = telemetry.copy()
grouped = enriched.groupby("machineID", group_keys=False)
for sensor in SENSOR_COLS:
    enriched[f"{sensor}_mean_6h"] = grouped[sensor].transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=6).mean()
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
    lambda s: s.shift(1).rolling(window=6, min_periods=6).sum()
)
enriched = enriched.drop(columns=["error_count"]).dropna().reset_index(drop=True)

telemetry_by_machine = {
    mid: grp.sort_values("datetime").reset_index(drop=True)
    for mid, grp in enriched.groupby("machineID")
}

# ── build RUL sequences — identical to build_component_rul_table +
#    build_component_sequences in training ────────────────────────────────────
print("Building RUL sequences...")
X_all, dt_all = [], []

for comp in COMPONENTS:
    comp_failures = failures.loc[failures["failure"].eq(comp)]
    comp_maint = maint.loc[maint["comp"].eq(comp)]
    rul_rows = []

    for mid, machine_fails in comp_failures.groupby("machineID"):
        machine_tel = telemetry.loc[telemetry["machineID"].eq(mid), ["datetime", "machineID"]]
        machine_maint = comp_maint.loc[comp_maint["machineID"].eq(mid)]

        for fail in machine_fails.sort_values("datetime").itertuples(index=False):
            fail_time = fail.datetime
            prior_maint = machine_maint.loc[machine_maint["datetime"].lt(fail_time)]
            last_maint = None if prior_maint.empty else prior_maint["datetime"].max()
            window_start = fail_time - pd.Timedelta(hours=MAX_RUL_HOURS)
            if last_maint is not None:
                window_start = max(window_start, last_maint)

            window = machine_tel.loc[
                machine_tel["datetime"].ge(window_start) &
                machine_tel["datetime"].lt(fail_time)
                ].copy()
            window["rul"] = ((fail_time - window["datetime"]).dt.total_seconds() / 3600).astype(int)
            rul_rows.append(
                window.loc[window["rul"].between(1, MAX_RUL_HOURS), ["datetime", "machineID", "rul"]]
            )

    if not rul_rows:
        continue

    rul_table = (
        pd.concat(rul_rows, ignore_index=True)
        .sort_values(["machineID", "datetime", "rul"])
        .drop_duplicates(["machineID", "datetime"], keep="first")
        .sort_values(["machineID", "datetime"])
        .reset_index(drop=True)
    )

    for row in rul_table.itertuples(index=False):
        machine_data = telemetry_by_machine.get(row.machineID)
        if machine_data is None:
            continue
        seq = machine_data.loc[machine_data["datetime"].le(row.datetime)].tail(SEQUENCE_LENGTH)
        if len(seq) != SEQUENCE_LENGTH:
            continue
        expected_start = row.datetime - pd.Timedelta(hours=SEQUENCE_LENGTH - 1)
        if seq["datetime"].iloc[0] != expected_start or seq["datetime"].iloc[-1] != row.datetime:
            continue
        X_all.append(seq[FEATURE_COLS].to_numpy(dtype=np.float32))
        dt_all.append(row.datetime)

X = np.stack(X_all).astype(np.float32)
dts = np.array(dt_all, dtype="datetime64[ns]")

# ── use only training split — identical to split_and_scale() ─────────────────
train_mask = (dts >= np.datetime64(TRAIN_START)) & (dts < np.datetime64(TRAIN_END))
X_train = X[train_mask]
print(f"Training sequences: {X_train.shape}  (used to compute mean/std)")

mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
std[std == 0] = 1.0

print("\nMean per feature:")
for name, val in zip(FEATURE_COLS, mean):
    print(f"  {name}: {val:.4f}")
print("\nStd per feature:")
for name, val in zip(FEATURE_COLS, std):
    print(f"  {name}: {val:.4f}")

# ── overwrite rul_norm_params.npz ────────────────────────────────────────────
out_path = MODEL_DIR / "rul_norm_params.npz"
np.savez(out_path, mean=mean, std=std, feature_names=np.array(FEATURE_COLS))
print(f"\nSaved to {out_path}")
print("Restart uvicorn to pick up the new norm params.")
