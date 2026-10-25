from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
RAW_DIR = BASE_DIR / "data" / "raw"


def describe_feature(features: pd.DataFrame, mask: pd.Series, column: str, label: str) -> None:
    values = features.loc[mask, column]
    percentiles = values.quantile([0.25, 0.50, 0.75, 0.95])
    print(f"{label} - {column}")
    print(f"mean: {values.mean():.6f}")
    print(f"max: {values.max():.6f}")
    print(f"p25: {percentiles.loc[0.25]:.6f}")
    print(f"p50: {percentiles.loc[0.50]:.6f}")
    print(f"p75: {percentiles.loc[0.75]:.6f}")
    print(f"p95: {percentiles.loc[0.95]:.6f}")


def investigation_1(features: pd.DataFrame) -> None:
    print("INVESTIGATION 1 - Actual error count values")
    positive = features["comp2_fail_24h"].eq(1)
    negative = features["comp2_fail_24h"].eq(0)

    for mask, label in [(positive, "comp2_fail_24h=1"), (negative, "comp2_fail_24h=0")]:
        for column in ["error2_count_24h", "error3_count_24h"]:
            describe_feature(features, mask, column, label)
        print()


def investigation_2(features: pd.DataFrame, errors: pd.DataFrame, failures: pd.DataFrame) -> None:
    print("INVESTIGATION 2 - Timing check")
    sample = features.loc[
        features["comp2_fail_24h"].eq(1) & features["error2_count_24h"].gt(0)
    ].sample(n=10, random_state=42)

    comp2_failures = failures.loc[failures["failure"].eq("comp2")]
    error2_events = errors.loc[errors["errorID"].eq("error2")]

    for row in sample.itertuples(index=False):
        window_start = row.datetime - pd.Timedelta(hours=24)
        window_errors = error2_events.loc[
            error2_events["machineID"].eq(row.machineID)
            & error2_events["datetime"].ge(window_start)
            & error2_events["datetime"].lt(row.datetime)
        ].sort_values("datetime")
        future_failure = comp2_failures.loc[
            comp2_failures["machineID"].eq(row.machineID)
            & comp2_failures["datetime"].gt(row.datetime)
            & comp2_failures["datetime"].le(row.datetime + pd.Timedelta(hours=24))
        ].sort_values("datetime").head(1)
        failure_time = None if future_failure.empty else future_failure.iloc[0]["datetime"]
        error_times = window_errors["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        errors_before_row = all(window_errors["datetime"].lt(row.datetime))
        errors_at_failure = False if failure_time is None else any(window_errors["datetime"].eq(failure_time))

        print(f"machineID={row.machineID}, row_datetime={row.datetime}")
        print(f"error2_count_24h={row.error2_count_24h}")
        print(f"actual_comp2_failure_datetime={failure_time}")
        print(f"error2_timestamps_before_row={error_times}")
        print(f"errors_before_row={errors_before_row}, errors_at_failure_timestamp={errors_at_failure}")
        print()


def investigation_3(features: pd.DataFrame) -> None:
    print("INVESTIGATION 3 - Cross check with 48h target")
    final_24h = features["comp2_fail_24h"].eq(1)
    prior_24_to_48h = features["comp2_fail_48h"].eq(1) & features["comp2_fail_24h"].eq(0)

    print("Rows in final 24h before comp2 failure:", int(final_24h.sum()))
    print("Rows in 24-48h before comp2 failure:", int(prior_24_to_48h.sum()))

    for column in ["error2_count_24h", "error3_count_24h"]:
        final_mean = features.loc[final_24h, column].mean()
        prior_mean = features.loc[prior_24_to_48h, column].mean()
        print(f"{column}: final_24h_mean={final_mean:.6f}, prior_24_to_48h_mean={prior_mean:.6f}")

    error2_final = features.loc[final_24h, "error2_count_24h"].mean()
    error2_prior = features.loc[prior_24_to_48h, "error2_count_24h"].mean()
    error3_final = features.loc[final_24h, "error3_count_24h"].mean()
    error3_prior = features.loc[prior_24_to_48h, "error3_count_24h"].mean()

    if error2_final > error2_prior and error3_final > error3_prior:
        print("Pattern: error counts are higher in final 24h than in the 24-48h window.")
        print("Conclusion: REAL SIGNAL")
    else:
        print("Pattern: error counts are similar or lower in final 24h than in the 24-48h window.")
        print("Conclusion: INVESTIGATE FURTHER")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])

    investigation_1(features)
    investigation_2(features, errors, failures)
    investigation_3(features)


if __name__ == "__main__":
    main()
