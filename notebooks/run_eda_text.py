from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def load_data() -> dict[str, pd.DataFrame]:
    paths = {path.stem: path for path in sorted(RAW_DIR.glob("*.csv"))}
    expected = [
        "PdM_telemetry",
        "PdM_errors",
        "PdM_maint",
        "PdM_failures",
        "PdM_machines",
    ]
    missing = [name for name in expected if name not in paths]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing}")

    return {
        "PdM_telemetry.csv": pd.read_csv(paths["PdM_telemetry"], parse_dates=["datetime"]),
        "PdM_errors.csv": pd.read_csv(paths["PdM_errors"], parse_dates=["datetime"]),
        "PdM_maint.csv": pd.read_csv(paths["PdM_maint"], parse_dates=["datetime"]),
        "PdM_failures.csv": pd.read_csv(paths["PdM_failures"], parse_dates=["datetime"]),
        "PdM_machines.csv": pd.read_csv(paths["PdM_machines"]),
    }


def main() -> None:
    frames = load_data()
    telemetry = frames["PdM_telemetry.csv"]
    errors = frames["PdM_errors.csv"]
    failures = frames["PdM_failures.csv"]
    machines = frames["PdM_machines.csv"]

    print("1. Unique machines in each file")
    for name, df in frames.items():
        print(f"{name}: {df['machineID'].nunique()}")

    print("\n2. Failures per component")
    failure_counts = failures["failure"].value_counts().reindex(
        ["comp1", "comp2", "comp3", "comp4"], fill_value=0
    )
    print(failure_counts.to_string())

    print("\n3. Errors per error type")
    error_counts = errors["errorID"].value_counts().reindex(
        ["error1", "error2", "error3", "error4", "error5"], fill_value=0
    )
    print(error_counts.to_string())

    print("\n4. Average number of errors per machine in windows before each failure")
    for hours in [24, 48, 72]:
        window_counts = []
        for row in failures[["machineID", "datetime"]].itertuples(index=False):
            start = row.datetime - pd.Timedelta(hours=hours)
            mask = (
                errors["machineID"].eq(row.machineID)
                & errors["datetime"].gt(start)
                & errors["datetime"].lt(row.datetime)
            )
            window_counts.append(int(mask.sum()))
        average = pd.Series(window_counts).mean()
        print(f"{hours}h before failure: {average:.4f} errors per failure window")

    print("\n5. Date range for each file")
    for name, df in frames.items():
        if "datetime" in df.columns:
            print(f"{name}: {df['datetime'].min()} to {df['datetime'].max()}")
        else:
            print(f"{name}: no datetime column")

    print("\n6. Machines by total failures")
    failures_per_machine = failures.groupby("machineID").size().reindex(
        machines["machineID"], fill_value=0
    )
    buckets = failures_per_machine.map(
        lambda count: (
            "0"
            if count == 0
            else "1"
            if count == 1
            else "2"
            if count == 2
            else "3"
            if count == 3
            else "4+"
        )
    )
    print(buckets.value_counts().reindex(["0", "1", "2", "3", "4+"], fill_value=0).to_string())

    print("\n7. Machine with the most failures")
    max_failures = int(failures_per_machine.max())
    max_machines = failures_per_machine[failures_per_machine.eq(max_failures)].index.tolist()
    print(f"Machine(s): {max_machines}; failures: {max_failures}")

    print("\n8. Average time between failures for machines that fail more than once")
    intervals_days = []
    sorted_failures = failures.sort_values(["machineID", "datetime"])
    for _, group in sorted_failures.groupby("machineID"):
        if len(group) > 1:
            intervals_days.extend(group["datetime"].diff().dropna().dt.total_seconds() / 86400)
    print(f"{pd.Series(intervals_days).mean():.4f} days")


if __name__ == "__main__":
    main()
