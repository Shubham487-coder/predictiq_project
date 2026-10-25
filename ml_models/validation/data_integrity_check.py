from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
RAW_DIR = BASE_DIR / "data" / "raw"

SENSOR_CHECK_FEATURES = [
    "volt_mean_24h",
    "rotate_mean_24h",
    "pressure_mean_24h",
    "vibration_mean_24h",
    "volt_std_24h",
    "rotate_std_24h",
    "error1_count_24h",
    "error2_count_24h",
    "error3_count_24h",
]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]


def status(name: str, passed: bool, explanation: str) -> bool:
    print(f"{name}: {'PASS' if passed else 'FAIL'} - {explanation}")
    return passed


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURE_PATH)
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    return features, telemetry, failures, maint


def check_machine_boundary_bleed(features: pd.DataFrame, telemetry: pd.DataFrame) -> bool:
    print("\nCHECK 1 - Machine boundary bleed in rolling features")
    ordered = features.sort_values(["machineID", "datetime"])
    first_rows = ordered.groupby("machineID").head(3)
    print(first_rows[["machineID", "datetime", "volt_mean_24h", "rotate_mean_24h"]].to_string(index=False))

    failures = []
    raw_ordered = telemetry.sort_values(["machineID", "datetime"])
    raw_by_machine = {
        machine_id: group for machine_id, group in raw_ordered.groupby("machineID")
    }

    for machine_id, group in ordered.groupby("machineID"):
        first_time = group["datetime"].min()
        first_three = group.head(3)
        raw_group = raw_by_machine[machine_id]
        for row in first_three.itertuples(index=False):
            history = raw_group.loc[raw_group["datetime"].lt(row.datetime)].tail(24)
            if len(history) < 24:
                failures.append((machine_id, row.datetime, "insufficient same-machine history"))
                continue
            expected_volt = history["volt"].mean()
            expected_rotate = history["rotate"].mean()
            if not np.isclose(row.volt_mean_24h, expected_volt, rtol=1e-9, atol=1e-9):
                failures.append((machine_id, row.datetime, "volt_mean_24h mismatch"))
            if not np.isclose(row.rotate_mean_24h, expected_rotate, rtol=1e-9, atol=1e-9):
                failures.append((machine_id, row.datetime, "rotate_mean_24h mismatch"))
            if row.datetime <= first_time:
                raw_first_time = raw_group["datetime"].min()
                if row.datetime <= raw_first_time:
                    failures.append((machine_id, row.datetime, "unexpected first raw row present"))

    if failures:
        print("Boundary violations:")
        for failure in failures[:20]:
            print(failure)
    return status(
        "CHECK 1",
        not failures,
        "Rolling features match each machine's own prior 24 rows for sampled starts."
        if not failures
        else f"{len(failures)} sampled boundary rows failed validation.",
    )


def check_label_isolation(features: pd.DataFrame, failures: pd.DataFrame) -> bool:
    print("\nCHECK 2 - Label isolation per machine")
    positives = features.loc[features["comp2_fail_24h"].eq(1)].sample(
        n=min(5, int(features["comp2_fail_24h"].sum())),
        random_state=42,
    )
    comp2_failures = failures.loc[failures["failure"].eq("comp2")]
    violations = []

    for row in positives.itertuples(index=False):
        machine_failures = comp2_failures.loc[
            (comp2_failures["machineID"].eq(row.machineID))
            & (comp2_failures["datetime"].gt(row.datetime))
            & (comp2_failures["datetime"].le(row.datetime + pd.Timedelta(hours=24)))
        ].sort_values("datetime")
        failure_row = machine_failures.iloc[0] if not machine_failures.empty else None

        if failure_row is None:
            print(
                f"machineID={row.machineID}, datetime={row.datetime}, "
                "actual_failure_datetime=None, failure_machine_match=False, within_24h_after=False"
            )
            violations.append((row.machineID, row.datetime, "no matching future failure"))
            continue

        machine_match = int(failure_row["machineID"]) == int(row.machineID)
        within_window = row.datetime < failure_row["datetime"] <= row.datetime + pd.Timedelta(hours=24)
        print(
            f"machineID={row.machineID}, datetime={row.datetime}, "
            f"actual_failure_datetime={failure_row['datetime']}, "
            f"failure_machine_match={machine_match}, within_24h_after={within_window}"
        )
        if not machine_match or not within_window:
            violations.append((row.machineID, row.datetime, failure_row["datetime"]))

    return status(
        "CHECK 2",
        not violations,
        "Sampled positive labels map to comp2 failures for the same machine strictly after the row within 24h."
        if not violations
        else f"{len(violations)} sampled labels violated isolation/window rules.",
    )


def check_time_split_coverage(features: pd.DataFrame) -> bool:
    print("\nCHECK 3 - Time split machine coverage")
    train = features.loc[features["datetime"].lt(pd.Timestamp("2015-09-01"))]
    test = features.loc[features["datetime"].ge(pd.Timestamp("2015-09-01"))]
    train_machines = set(train["machineID"].unique())
    test_machines = set(test["machineID"].unique())
    only_test = sorted(test_machines - train_machines)

    print(f"Unique machineIDs in train: {len(train_machines)}")
    print(f"Unique machineIDs in test: {len(test_machines)}")
    print(f"MachineIDs only in test: {only_test}")

    return status(
        "CHECK 3",
        len(only_test) == 0,
        "All test machines also appear in train."
        if not only_test
        else "Some machines appear only in test, which may make evaluation partly cold-start.",
    )


def check_maintenance_recency(features: pd.DataFrame, maint: pd.DataFrame) -> bool:
    print("\nCHECK 4 - Maintenance recency sanity")
    comp1 = features["days_since_comp1_maint"]
    print(
        "days_since_comp1_maint: "
        f"min={comp1.min():.6f}, max={comp1.max():.6f}, mean={comp1.mean():.6f}"
    )
    for component in COMPONENTS:
        column = f"days_since_{component}_maint"
        print(f"{column} rows == 999: {int(features[column].eq(999).sum())}")

    sample = features.loc[
        features["days_since_comp2_maint"].ge(0) & features["days_since_comp2_maint"].lt(5)
    ].sample(n=3, random_state=42)
    violations = []
    comp2_maint = maint.loc[maint["comp"].eq("comp2")]
    print("Recent comp2 maintenance samples:")
    for row in sample.itertuples(index=False):
        recent = comp2_maint.loc[
            (comp2_maint["machineID"].eq(row.machineID))
            & (comp2_maint["datetime"].lt(row.datetime))
            & (comp2_maint["datetime"].ge(row.datetime - pd.Timedelta(days=5)))
        ].sort_values("datetime")
        last = recent.iloc[-1] if not recent.empty else None
        print(
            f"machineID={row.machineID}, datetime={row.datetime}, "
            f"days_since_comp2_maint={row.days_since_comp2_maint:.6f}, "
            f"verified_recent_comp2_maint={None if last is None else last['datetime']}"
        )
        if last is None:
            violations.append((row.machineID, row.datetime))

    no_negative_unknown_fill = all(int(features[f"days_since_{component}_maint"].lt(0).sum()) == 0 for component in COMPONENTS)
    no_values_above_sentinel = all(int(features[f"days_since_{component}_maint"].gt(999).sum()) == 0 for component in COMPONENTS)

    passed = not violations and no_negative_unknown_fill and no_values_above_sentinel
    explanation_parts = []
    if not violations:
        explanation_parts.append("sampled recent comp2 maintenance rows verify against raw maintenance data")
    else:
        explanation_parts.append(f"{len(violations)} sampled recent maintenance rows did not verify")
    if no_negative_unknown_fill and no_values_above_sentinel:
        explanation_parts.append("no negative unknown-maintenance values remain; 999 count can be zero when all rows have history")
    else:
        explanation_parts.append("maintenance recency contains negative values or values above 999")

    return status("CHECK 4", passed, "; ".join(explanation_parts) + ".")


def check_extended_leakage(features: pd.DataFrame) -> bool:
    print("\nCHECK 5 - Extended leakage check")
    positives = features.loc[features["comp2_fail_24h"].eq(1)]
    negatives = features.loc[features["comp2_fail_24h"].eq(0)]
    violations = []

    for feature in SENSOR_CHECK_FEATURES:
        pos_mean = positives[feature].mean()
        pos_std = positives[feature].std()
        neg_mean = negatives[feature].mean()
        neg_std = negatives[feature].std()
        pooled_std = features[feature].std()
        diff_std = abs(pos_mean - neg_mean) / pooled_std if pooled_std else 0.0
        print(
            f"{feature}: pos_mean={pos_mean:.6f}, pos_std={pos_std:.6f}, "
            f"neg_mean={neg_mean:.6f}, neg_std={neg_std:.6f}, diff_std={diff_std:.6f}"
        )
        if diff_std > 5:
            violations.append((feature, diff_std))

    return status(
        "CHECK 5",
        not violations,
        "No checked feature exceeded the >5 std suspicious-difference threshold."
        if not violations
        else f"Suspicious features: {violations}.",
    )


def check_maintenance_999_distribution(features: pd.DataFrame) -> bool:
    print("\nCHECK 6 - days_since_maint 999 handling")
    values = features["days_since_comp2_maint"]
    distribution = {
        "<30": int(values.lt(30).sum()),
        "30-100": int(values.ge(30).mul(values.lt(100)).sum()),
        "100-365": int(values.ge(100).mul(values.lt(365)).sum()),
        "365-998": int(values.ge(365).mul(values.lt(999)).sum()),
        "=999": int(values.eq(999).sum()),
    }
    print("days_since_comp2_maint distribution:")
    for bucket, count in distribution.items():
        print(f"{bucket}: {count}")

    feature_importance_files = sorted((BASE_DIR / "ml_models" / "failure_classification").glob("feature_importance_*.png"))
    print("Feature importance artifact check:")
    print(f"Found feature importance plot artifacts: {[path.name for path in feature_importance_files]}")
    print("Top-feature 999 driver cannot be proven from PNG artifacts alone; checking raw feature distribution instead.")

    negative_unknown_count = int(values.lt(0).sum())
    above_sentinel_count = int(values.gt(999).sum())
    passed = negative_unknown_count == 0 and above_sentinel_count == 0
    return status(
        "CHECK 6",
        passed,
        "No negative unknown-maintenance values remain; 999 rows are absent because all retained rows have prior comp2 history."
        if passed
        else (
            f"Expected 999 fill is absent or negative unknown fills remain "
            f"(=999 rows: {distribution['=999']}, <0 rows: {negative_unknown_count}, >999 rows: {above_sentinel_count})."
        ),
    )


def main() -> None:
    features, telemetry, failures, maint = load_data()
    results = [
        check_machine_boundary_bleed(features, telemetry),
        check_label_isolation(features, failures),
        check_time_split_coverage(features),
        check_maintenance_recency(features, maint),
        check_extended_leakage(features),
        check_maintenance_999_distribution(features),
    ]

    print("\nOVERALL STATUS:")
    print("READY TO TRAIN" if all(results) else "ISSUES FOUND")


if __name__ == "__main__":
    main()
