"""
test_accuracy.py — PredictIQ accuracy evaluation via API.

Replicates the exact evaluation from training:
  - Classifier: same time-based test split (2015-09-01 to 2016-01-01),
                same tune_threshold() logic, same metrics
  - RUL:        same sequence builder, same evaluate_component() logic,
                same metrics (MAE, RMSE, R2, within_1_day, per-day accuracy)

All predictions go through the running API (http://localhost:8000).
Start uvicorn before running this script.

Usage:
    python test_accuracy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

BASE_URL = "http://localhost:8000/api/v1"
BASE_DIR = Path(__file__).resolve().parent

# ── Same splits as training ──────────────────────────────────────────────────
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")

# ── Same threshold sweep as training ────────────────────────────────────────
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
COMPONENT_TO_INDEX = {c: i for i, c in enumerate(COMPONENTS)}

INT_COLS = {"age", "model_model1", "model_model2", "model_model3", "model_model4"}

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

RUL_FEATURE_COLS = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_6h", "rotate_mean_6h", "pressure_mean_6h", "vibration_mean_6h",
    "error_count_6h",
]

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]

SEQUENCE_LENGTH = 48
MAX_RUL_HOURS = 100
MAX_DAY = 5
MASK_VALUE = -1.0

# ── Reference metrics from training ─────────────────────────────────────────
REF_CLASSIFIER = {
    "comp1": {"f1": 0.963292, "recall": 0.975490},
    "comp2": {"f1": 0.981533, "recall": 0.981533},
    "comp3": {"f1": 0.969287, "recall": 0.969792},
    "comp4": {"f1": 0.981523, "recall": 0.983704},
}
REF_RUL = {
    "comp1": {"mae": 10.11, "r2": 0.761, "within_1_day": 0.9696},
    "comp2": {"mae": 9.55, "r2": 0.773, "within_1_day": 0.9677},
    "comp3": {"mae": 8.70, "r2": 0.802, "within_1_day": 0.9832},
    "comp4": {"mae": 8.71, "r2": 0.803, "within_1_day": 0.9822},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    """Exact replica of tune_threshold() from train_all_components.py."""
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append((threshold, precision, recall, f1))
    candidates = [row for row in rows if row[2] >= 0.85]
    if candidates:
        return max(candidates, key=lambda r: r[3])
    return max(rows, key=lambda r: r[3])


def hours_to_days(hours: np.ndarray) -> np.ndarray:
    """Exact replica of hours_to_days() from train_rul_multihead_v2.py."""
    return np.ceil(np.clip(hours, 1, MAX_RUL_HOURS) / 24).clip(1, MAX_DAY).astype(np.int64)


def row_to_payload(row: pd.Series, machine_id: int) -> dict:
    payload = {"machine_id": machine_id, "alert_threshold": 0.5}
    for col in CLASSIFIER_FEATURES:
        payload[col] = int(row[col]) if col in INT_COLS else float(row[col])
    return payload


# ── Test 5: Classifier accuracy ───────────────────────────────────────────────

def test_classifier_accuracy(client: httpx.Client) -> None:
    print("=" * 70)
    print("TEST 5: Classifier accuracy (via API, same split as training)")
    print("=" * 70)

    features_path = BASE_DIR / "data" / "processed" / "features.parquet"
    print(f"Loading {features_path} ...")
    df = pd.read_parquet(features_path).sort_values("datetime").reset_index(drop=True)

    label_cols = [f"{c}_fail_24h" for c in COMPONENTS]
    test_mask = (df["datetime"] >= TEST_START) & (df["datetime"] < TEST_END)
    test_df = df.loc[test_mask].dropna(subset=CLASSIFIER_FEATURES + label_cols).reset_index(drop=True)

    print(f"Test rows: {len(test_df)}")
    print(f"Sending {len(test_df)} rows to POST /classify in batches ...\n")

    BATCH = 50
    all_proba: dict[str, list[float]] = {c: [] for c in COMPONENTS}

    for start in range(0, len(test_df), BATCH):
        chunk = test_df.iloc[start:start + BATCH]
        payload = {
            "requests": [
                row_to_payload(row, machine_id=int(row["machineID"]))
                for _, row in chunk.iterrows()
            ]
        }
        resp = client.post(f"{BASE_URL}/classify/batch", json=payload, timeout=120)
        resp.raise_for_status()
        results = resp.json()["data"]
        for result in results:
            for comp in COMPONENTS:
                all_proba[comp].append(result[comp]["failure_probability"])

        if (start // BATCH) % 10 == 0:
            print(f"  ... {start + len(chunk)}/{len(test_df)} rows done")

    print()

    summary_rows = []
    for comp in COMPONENTS:
        y_true = test_df[f"{comp}_fail_24h"].to_numpy()
        y_proba = np.array(all_proba[comp])
        threshold, precision, recall, f1 = tune_threshold(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else float("nan")

        ref_f1 = REF_CLASSIFIER[comp]["f1"]
        ref_recall = REF_CLASSIFIER[comp]["recall"]
        delta_f1 = f1 - ref_f1

        summary_rows.append({
            "component": comp,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "ref_f1": ref_f1,
            "delta_f1": delta_f1,
        })

        status = "✓" if abs(delta_f1) < 0.005 else ("↑" if delta_f1 > 0 else "↓")
        print(f"{comp}: threshold={threshold:.2f}  precision={precision:.4f}  "
              f"recall={recall:.4f}  f1={f1:.4f}  roc_auc={roc_auc:.4f}  "
              f"[ref_f1={ref_f1:.4f}  Δ={delta_f1:+.4f} {status}]")

    print()
    summary = pd.DataFrame(summary_rows)
    print(summary[["component", "threshold", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print()


# ── Test 6: RUL accuracy ─────────────────────────────────────────────────────

def build_rul_test_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild test sequences exactly as in training:
    enrich_telemetry → build_component_rul_table → build_component_sequences → test split.
    Returns X_test (N,48,9), y_test (N,), component_test (N,) for the test period.
    """
    raw_dir = BASE_DIR / "data" / "raw"
    print("Loading raw CSVs for RUL sequence builder ...")
    telemetry = pd.read_csv(raw_dir / "PdM_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv(raw_dir / "PdM_failures.csv", parse_dates=["datetime"])
    maint = pd.read_csv(raw_dir / "PdM_maint.csv", parse_dates=["datetime"])
    errors = pd.read_csv(raw_dir / "PdM_errors.csv", parse_dates=["datetime"])

    for df in [telemetry, failures, maint, errors]:
        df.sort_values(["machineID", "datetime"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # enrich_telemetry — same as training
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

    # Build per-machine lookup from ENRICHED telemetry (same as training)
    telemetry_by_machine = {
        mid: grp.sort_values("datetime").reset_index(drop=True)
        for mid, grp in enriched.groupby("machineID")
    }

    X_all, y_all, dt_all, comp_all = [], [], [], []

    for comp in COMPONENTS:
        comp_idx = COMPONENT_TO_INDEX[comp]
        comp_failures = failures.loc[failures["failure"].eq(comp)]
        comp_maint = maint.loc[maint["comp"].eq(comp)]
        rul_rows = []

        for mid, machine_fails in comp_failures.groupby("machineID"):
            # FIX: use enriched telemetry (telemetry_by_machine) for window lookup,
            # not raw telemetry — same as training
            machine_tel = telemetry_by_machine.get(mid)
            if machine_tel is None:
                continue
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

            X_all.append(seq[RUL_FEATURE_COLS].to_numpy(dtype=np.float32))
            y_all.append(float(row.rul))
            dt_all.append(row.datetime)
            comp_all.append(comp_idx)

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.float32)
    dts = np.array(dt_all, dtype="datetime64[ns]")
    ci = np.array(comp_all, dtype=np.int64)

    test_mask = (dts >= np.datetime64(TEST_START)) & (dts < np.datetime64(TEST_END))
    return X[test_mask], y[test_mask], ci[test_mask]


def test_rul_accuracy(client: httpx.Client) -> None:
    print("=" * 70)
    print("TEST 6: RUL accuracy (via API, same split as training)")
    print("=" * 70)

    X_test, y_test, comp_test = build_rul_test_sequences()
    print(f"Test sequences: {len(X_test)}\n")
    print(f"Sending {len(X_test)} sequences to POST /rul/batch in batches ...\n")

    BATCH = 20
    y_pred_all = []

    for start in range(0, len(X_test), BATCH):
        X_chunk = X_test[start:start + BATCH]
        comp_chunk = comp_test[start:start + BATCH]
        comp_names = [COMPONENTS[i] for i in comp_chunk]

        payload = {
            "requests": [
                {
                    "machine_id": 1,
                    "component": comp_names[i],
                    "sensor_sequence": X_chunk[i].tolist(),
                }
                for i in range(len(X_chunk))
            ]
        }
        resp = client.post(f"{BASE_URL}/rul/batch", json=payload, timeout=120)
        resp.raise_for_status()
        results = resp.json()["data"]
        for r in results:
            y_pred_all.append(r["predicted_rul_hours"])

        if (start // BATCH) % 25 == 0:
            print(f"  ... {start + len(X_chunk)}/{len(X_test)} sequences done")

    y_pred = np.array(y_pred_all, dtype=np.float32)
    print()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for comp in COMPONENTS:
        comp_idx = COMPONENT_TO_INDEX[comp]
        mask = comp_test == comp_idx
        y_true_c = y_test[mask]
        y_pred_c = np.clip(y_pred[mask], 0, MAX_RUL_HOURS)

        mae = mean_absolute_error(y_true_c, y_pred_c)
        rmse = float(np.sqrt(mean_squared_error(y_true_c, y_pred_c)))
        r2 = r2_score(y_true_c, y_pred_c)
        true_day = hours_to_days(y_true_c)
        pred_day = hours_to_days(y_pred_c)
        within_1d = float(np.mean(np.abs(pred_day - true_day) <= 1))

        ref = REF_RUL[comp]
        print(f"{comp}:")
        print(f"  MAE={mae:.2f}h  (ref {ref['mae']:.2f}h)  "
              f"RMSE={rmse:.2f}h  R²={r2:.3f}  (ref {ref['r2']:.3f})")
        print(f"  within_1_day={within_1d:.4f}  (ref {ref['within_1_day']:.4f})")
        for day in range(1, MAX_DAY + 1):
            day_mask = true_day == day
            if day_mask.any():
                acc = float(np.mean(pred_day[day_mask] == true_day[day_mask]))
                print(f"  day{day}_accuracy={acc:.4f}  (n={day_mask.sum()})")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("PredictIQ — Accuracy evaluation via API")
    print(f"Test period: {TEST_START.date()} → {TEST_END.date()}\n")

    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"API healthy — models_loaded={r.json()['data']['models_loaded']}\n")
    except Exception as e:
        print(f"✗ API not reachable at {BASE_URL}: {e}")
        print("  Start it with: uvicorn api.main:app --reload")
        sys.exit(1)

    with httpx.Client() as client:
        test_classifier_accuracy(client)
        test_rul_accuracy(client)

    print("Done.")


if __name__ == "__main__":
    main()
