from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features.parquet"

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
ERROR_IDS = ["error1", "error2", "error3", "error4", "error5"]
HORIZONS = [24, 48, 72]
TELEMETRY_COLS = ["volt", "rotate", "pressure", "vibration"]


def load_csvs() -> dict[str, pd.DataFrame]:
    data = {
        "telemetry": pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"]),
        "errors": pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"]),
        "maint": pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"]),
        "failures": pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"]),
        "machines": pd.read_csv(RAW_DIR / "PdM_machines.csv"),
    }

    data["telemetry"] = data["telemetry"].sort_values(["machineID", "datetime"]).reset_index(drop=True)
    data["errors"] = data["errors"].sort_values(["machineID", "datetime"]).reset_index(drop=True)
    data["maint"] = data["maint"].sort_values(["machineID", "datetime"]).reset_index(drop=True)
    data["failures"] = data["failures"].sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return data


def add_failure_labels(features: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    base = features[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()

    for component in COMPONENTS:
        component_failures = (
            failures.loc[failures["failure"].eq(component), ["machineID", "datetime"]]
            .rename(columns={"datetime": "failure_datetime"})
            .sort_values(["failure_datetime", "machineID"])
        )

        labeled = pd.merge_asof(
            base,
            component_failures,
            left_on="datetime",
            right_on="failure_datetime",
            by="machineID",
            direction="forward",
            allow_exact_matches=False,
        ).set_index("index")

        hours_to_failure = (
            labeled["failure_datetime"] - labeled["datetime"]
        ).dt.total_seconds() / 3600

        for horizon in HORIZONS:
            features[f"{component}_fail_{horizon}h"] = (
                hours_to_failure.le(horizon).fillna(False).astype("int8").sort_index().to_numpy()
            )

    return features


def add_telemetry_rollups(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby("machineID", group_keys=False)
    for hours in HORIZONS:
        for column in TELEMETRY_COLS:
            shifted = grouped[column].shift(1)
            rolling = shifted.groupby(features["machineID"]).rolling(
                window=hours,
                min_periods=hours,
            )
            features[f"{column}_mean_{hours}h"] = rolling.mean().reset_index(level=0, drop=True)
            features[f"{column}_std_{hours}h"] = (
                rolling.std().reset_index(level=0, drop=True)
            )
    return features


def add_error_rollups(features: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    error_counts = (
        pd.crosstab([errors["machineID"], errors["datetime"]], errors["errorID"])
        .reindex(columns=ERROR_IDS, fill_value=0)
        .reset_index()
    )

    features = features.merge(error_counts, on=["machineID", "datetime"], how="left")
    features[ERROR_IDS] = features[ERROR_IDS].fillna(0).astype("int16")

    grouped = features.groupby("machineID", group_keys=False)
    for hours in HORIZONS:
        for error_id in ERROR_IDS:
            shifted = grouped[error_id].shift(1)
            features[f"{error_id}_count_{hours}h"] = (
                shifted.groupby(features["machineID"])
                .rolling(window=hours, min_periods=hours)
                .sum()
                .reset_index(level=0, drop=True)
            )

    return features.drop(columns=ERROR_IDS)


def add_maintenance_recency(features: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    asof_base = features[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()

    for component in COMPONENTS:
        component_maint = (
            maint.loc[maint["comp"].eq(component), ["machineID", "datetime"]]
            .rename(columns={"datetime": "last_maint_datetime"})
            .sort_values(["last_maint_datetime", "machineID"])
        )

        merged = pd.merge_asof(
            asof_base,
            component_maint,
            left_on="datetime",
            right_on="last_maint_datetime",
            by="machineID",
            direction="backward",
            allow_exact_matches=False,
        ).set_index("index")

        days_since = (
            merged["datetime"] - merged["last_maint_datetime"]
        ).dt.total_seconds() / 86400
        features[f"days_since_{component}_maint"] = days_since.sort_index().to_numpy()

    return features


def add_machine_metadata(features: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    machines_encoded = pd.get_dummies(
        machines[["machineID", "model", "age"]],
        columns=["model"],
        prefix="model",
        dtype="int8",
    )
    return features.merge(machines_encoded, on="machineID", how="left")


def add_interaction_features(features: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    features["volt_rotate_ratio_24h"] = (
        features["volt_mean_24h"] / features["rotate_mean_24h"].replace(0, np.nan)
    )
    features["vibration_cv_24h"] = (
        features["vibration_std_24h"] / features["vibration_mean_24h"].replace(0, np.nan)
    )
    features["error_count_total_72h"] = features[
        [
            "error1_count_72h",
            "error2_count_72h",
            "error3_count_72h",
            "error4_count_72h",
            "error5_count_72h",
        ]
    ].sum(axis=1)

    asof_base = features[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()
    error_events = (
        errors[["machineID", "datetime"]]
        .rename(columns={"datetime": "last_error_datetime"})
        .sort_values(["last_error_datetime", "machineID"])
    )
    last_errors = pd.merge_asof(
        asof_base,
        error_events,
        left_on="datetime",
        right_on="last_error_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    features["hours_since_last_error"] = (
        (last_errors["datetime"] - last_errors["last_error_datetime"]).dt.total_seconds() / 3600
    ).sort_index().to_numpy()
    return features


def add_leakage_check_columns(features: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    base = features[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()
    comp2_failures = (
        failures.loc[failures["failure"].eq("comp2"), ["machineID", "datetime"]]
        .rename(columns={"datetime": "next_comp2_failure_datetime"})
        .sort_values(["next_comp2_failure_datetime", "machineID"])
    )
    next_failures = pd.merge_asof(
        base,
        comp2_failures,
        left_on="datetime",
        right_on="next_comp2_failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    ).set_index("index")
    features["_next_comp2_failure_datetime"] = (
        next_failures["next_comp2_failure_datetime"].sort_index().to_numpy()
    )
    return features


def print_leakage_check(features: pd.DataFrame) -> None:
    print("\nLeakage check: comp2_fail_24h positive samples")
    positives = features.loc[features["comp2_fail_24h"].eq(1)]
    positive_sample = positives.sample(n=min(10, len(positives)), random_state=42)
    for _, row in positive_sample.iterrows():
        is_after = row["_next_comp2_failure_datetime"] > row["datetime"]
        print(
            f"machineID={row['machineID']}, datetime={row['datetime']}, "
            f"actual_failure_datetime={row['_next_comp2_failure_datetime']}, "
            f"failure_after_row={is_after}"
        )

    pos_volt = positive_sample["volt_mean_24h"]
    negative_sample = features.loc[features["comp2_fail_24h"].eq(0)].sample(
        n=min(10, int(features["comp2_fail_24h"].eq(0).sum())),
        random_state=42,
    )
    neg_volt = negative_sample["volt_mean_24h"]
    population_std = features["volt_mean_24h"].std()
    std_delta = abs(pos_volt.mean() - neg_volt.mean()) / population_std

    print("\nLeakage check: volt_mean_24h comparison")
    print(f"Positive sample mean volt_mean_24h: {pos_volt.mean():.6f}")
    print(f"Negative sample mean volt_mean_24h: {neg_volt.mean():.6f}")
    print(f"Difference in population std units: {std_delta:.6f}")
    if std_delta > 3:
        print("Suspicious difference detected (>3 std).")
    else:
        print("No suspicious >3 std difference detected.")


def build_features() -> tuple[pd.DataFrame, dict[str, int], int, int]:
    data = load_csvs()
    features = data["telemetry"].copy()

    features = add_failure_labels(features, data["failures"])
    features = add_telemetry_rollups(features)
    features = add_error_rollups(features, data["errors"])
    features = add_maintenance_recency(features, data["maint"])
    features = add_machine_metadata(features, data["machines"])
    features = add_interaction_features(features, data["errors"])

    nan_counts_before = features.isna().sum()
    nan_counts_before = nan_counts_before[nan_counts_before.gt(0)].to_dict()
    nan_rows_before = int(features.isna().any(axis=1).sum())

    # No previous maintenance exists for some early telemetry rows; 999 means unknown/no prior replacement.
    maintenance_cols = [f"days_since_{component}_maint" for component in COMPONENTS]
    features[maintenance_cols] = features[maintenance_cols].fillna(999.0)
    features["hours_since_last_error"] = features["hours_since_last_error"].fillna(9999.0)

    remaining_nan = int(features.isna().sum().sum())
    if remaining_nan:
        features = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    else:
        features = features.replace([np.inf, -np.inf], np.nan)

    rows_after_nan_handling = len(features)
    leakage_features = add_leakage_check_columns(features.copy(), data["failures"])
    print_leakage_check(leakage_features)

    return features, nan_counts_before, nan_rows_before, rows_after_nan_handling


def print_summary(
    features: pd.DataFrame,
    nan_counts_before: dict[str, int],
    nan_rows_before: int,
    rows_after_nan_handling: int,
) -> None:
    label_cols = [
        f"{component}_fail_{horizon}h"
        for horizon in HORIZONS
        for component in COMPONENTS
    ]

    print(f"Final table shape: {features.shape}")
    print("Columns:")
    print(list(features.columns))

    print("\nLabel distribution:")
    for column in label_cols:
        counts = features[column].value_counts().reindex([0, 1], fill_value=0)
        print(f"{column}: 0={counts[0]}, 1={counts[1]}")

    print("\nNaN handling:")
    if nan_counts_before:
        print(f"Rows with NaN before handling: {nan_rows_before}")
        print("NaN counts before handling:")
        for column, count in nan_counts_before.items():
            print(f"{column}: {count}")
        print("Filled maintenance recency NaNs with 999.0 to indicate no prior replacement.")
        print("Dropped rows with incomplete rolling telemetry/error windows.")
    else:
        print("No NaN values found before handling.")

    remaining_nan_rows = int(features.isna().any(axis=1).sum())
    print(f"Rows after NaN handling: {rows_after_nan_handling}")
    print(f"Rows with NaN after handling: {remaining_nan_rows}")


def main() -> None:
    features, nan_counts_before, nan_rows_before, rows_after_nan_handling = build_features()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved features to: {OUTPUT_PATH}")
    print_summary(features, nan_counts_before, nan_rows_before, rows_after_nan_handling)


if __name__ == "__main__":
    main()
