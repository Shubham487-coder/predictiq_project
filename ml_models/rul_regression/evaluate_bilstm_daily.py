from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = BASE_DIR / "ml_models" / "rul_regression" / "rul_model_comp2_bilstm_att.pt"

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
COMPONENT = "comp2"
SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
FEATURE_COLS = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
    "volt_mean_6h",
    "rotate_mean_6h",
    "pressure_mean_6h",
    "vibration_mean_6h",
    "error_count_6h",
]
SEQUENCE_LENGTH = 48
MAX_RUL_HOURS = 100
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
BATCH_SIZE = 256
HIDDEN_SIZE = 128
MAX_DAY = 5
ORIGINAL_ACC_24H = 0.916
DAILY_CLASSIFICATION_WITHIN_1_DAY = 0.782
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class RULBiLSTMAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attention = BahdanauAttention(hidden_size=HIDDEN_SIZE)
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context, _ = self.attention(lstm_output)
        return self.output(context).squeeze(-1)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    maint = maint.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return telemetry, failures, maint, errors


def enrich_telemetry(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    enriched = telemetry.copy()
    grouped = enriched.groupby("machineID", group_keys=False)
    for sensor in SENSOR_COLS:
        enriched[f"{sensor}_mean_6h"] = (
            grouped[sensor]
            .rolling(window=6, min_periods=6)
            .mean()
            .reset_index(level=0, drop=True)
        )

    error_counts = (
        errors.assign(error_count=1)
        .groupby(["machineID", "datetime"])["error_count"]
        .sum()
        .reset_index()
    )
    enriched = enriched.merge(error_counts, on=["machineID", "datetime"], how="left")
    enriched["error_count"] = enriched["error_count"].fillna(0)
    enriched["error_count_6h"] = (
        enriched.groupby("machineID")["error_count"]
        .rolling(window=6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return enriched.drop(columns=["error_count"]).dropna().reset_index(drop=True)


def build_component_rul_table(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    rows = []
    component_failures = failures.loc[failures["failure"].eq(component)]
    component_maint = maint.loc[maint["comp"].eq(component)]

    for machine_id, machine_failures in component_failures.groupby("machineID"):
        machine_telemetry = telemetry.loc[
            telemetry["machineID"].eq(machine_id), ["datetime", "machineID"]
        ]
        machine_maint = component_maint.loc[component_maint["machineID"].eq(machine_id)]
        for failure in machine_failures.sort_values("datetime").itertuples(index=False):
            failure_time = failure.datetime
            prior_maint = machine_maint.loc[machine_maint["datetime"].lt(failure_time)]
            last_maint_time = None if prior_maint.empty else prior_maint["datetime"].max()
            window_start = failure_time - pd.Timedelta(hours=MAX_RUL_HOURS)
            if last_maint_time is not None:
                window_start = max(window_start, last_maint_time)

            window = machine_telemetry.loc[
                machine_telemetry["datetime"].ge(window_start)
                & machine_telemetry["datetime"].lt(failure_time)
            ].copy()
            window["rul"] = ((failure_time - window["datetime"]).dt.total_seconds() / 3600).astype(int)
            rows.append(
                window.loc[
                    window["rul"].between(1, MAX_RUL_HOURS),
                    ["datetime", "machineID", "rul"],
                ]
            )

    if not rows:
        return pd.DataFrame(columns=["datetime", "machineID", "rul"])
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["machineID", "datetime", "rul"])
        .drop_duplicates(["machineID", "datetime"], keep="first")
        .sort_values(["machineID", "datetime"])
        .reset_index(drop=True)
    )


def build_all_rul_tables(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables = {}
    print("RUL row counts by component:", flush=True)
    for component in COMPONENTS:
        table = build_component_rul_table(telemetry, failures, maint, component)
        tables[component] = table
        print(f"{component}: {len(table)}", flush=True)
    return tables


def build_sequences(
    telemetry_features: pd.DataFrame,
    rul_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    telemetry_by_machine = {
        machine_id: group.sort_values("datetime").reset_index(drop=True)
        for machine_id, group in telemetry_features.groupby("machineID")
    }
    X_rows = []
    y_rows = []
    datetime_rows = []

    for row in rul_table.itertuples(index=False):
        machine_data = telemetry_by_machine.get(row.machineID)
        if machine_data is None:
            continue

        sequence = machine_data.loc[machine_data["datetime"].le(row.datetime)].tail(SEQUENCE_LENGTH)
        if len(sequence) != SEQUENCE_LENGTH:
            continue

        expected_start = row.datetime - pd.Timedelta(hours=SEQUENCE_LENGTH - 1)
        if sequence["datetime"].iloc[0] != expected_start or sequence["datetime"].iloc[-1] != row.datetime:
            continue

        X_rows.append(sequence[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_rows.append(float(row.rul))
        datetime_rows.append(row.datetime)

    return (
        np.stack(X_rows).astype(np.float32),
        np.array(y_rows, dtype=np.float32),
        np.array(datetime_rows, dtype="datetime64[ns]"),
    )


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    datetimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = (datetimes >= np.datetime64(TRAIN_START)) & (datetimes < np.datetime64(TRAIN_END))
    test_mask = (datetimes >= np.datetime64(TEST_START)) & (datetimes < np.datetime64(TEST_END))

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0

    return ((X_test - mean) / std).astype(np.float32), y_test


def load_model() -> RULBiLSTMAttention:
    model = RULBiLSTMAttention().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: RULBiLSTMAttention, X: np.ndarray) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")
    predictions = []

    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            predictions.append(model(batch_X).detach().cpu().numpy())

    return np.concatenate(predictions)


def hours_to_days(hours: np.ndarray) -> np.ndarray:
    clipped_hours = np.clip(hours, 1, MAX_RUL_HOURS)
    return np.ceil(clipped_hours / 24).clip(1, MAX_DAY).astype(np.int64)


def calculate_day_metrics(
    y_true_hours: np.ndarray,
    y_pred_hours: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    true_day = hours_to_days(y_true_hours)
    predicted_day = hours_to_days(y_pred_hours)

    metrics = {
        "exact_day_accuracy": float(np.mean(predicted_day == true_day)),
        "within_1_day_accuracy": float(np.mean(np.abs(predicted_day - true_day) <= 1)),
    }

    for day in range(1, MAX_DAY + 1):
        mask = true_day == day
        metrics[f"day_{day}_accuracy"] = (
            float(np.mean(predicted_day[mask] == true_day[mask])) if mask.any() else float("nan")
        )
        metrics[f"day_{day}_support"] = int(mask.sum())

    matrix = confusion_matrix(true_day, predicted_day, labels=np.arange(1, MAX_DAY + 1))
    return metrics, matrix


def format_percent(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def print_comparison_table(metrics: dict[str, float]) -> None:
    rows = [
        ("Original acc_24h (hourly)", format_percent(ORIGINAL_ACC_24H)),
        ("Within-1-day (daily buckets)", format_percent(metrics["within_1_day_accuracy"])),
        ("Daily classification model", format_percent(DAILY_CLASSIFICATION_WITHIN_1_DAY)),
        ("Exact day match accuracy", format_percent(metrics["exact_day_accuracy"])),
        ("Day 1 accuracy", format_percent(metrics["day_1_accuracy"])),
        ("Day 2 accuracy", format_percent(metrics["day_2_accuracy"])),
        ("Day 3 accuracy", format_percent(metrics["day_3_accuracy"])),
        ("Day 4 accuracy", format_percent(metrics["day_4_accuracy"])),
        ("Day 5 accuracy", format_percent(metrics["day_5_accuracy"])),
    ]

    print("\nMetric                          | Value")
    print("--------------------------------|--------")
    for label, value in rows:
        print(f"{label:<31} | {value}")


def print_per_day_support(metrics: dict[str, float]) -> None:
    print("\nPer-day support:")
    for day in range(1, MAX_DAY + 1):
        print(f"Day {day}: {int(metrics[f'day_{day}_support'])}")


def print_confusion_matrix(matrix: np.ndarray) -> None:
    labels = [f"Pred {day}" for day in range(1, MAX_DAY + 1)]
    print("\nConfusion matrix, rows=true day and columns=predicted day:")
    print("          " + " ".join(f"{label:>8}" for label in labels))
    for day, row in enumerate(matrix, start=1):
        print(f"True {day:<4} " + " ".join(f"{value:8d}" for value in row))


def main() -> None:
    print("Evaluating BiLSTM-Attention model with daily RUL buckets", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    telemetry, failures, maint, errors = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors)
    rul_tables = build_all_rul_tables(telemetry, failures, maint)
    X, y, datetimes = build_sequences(telemetry_features, rul_tables[COMPONENT])
    X_test, y_test = split_and_scale(X, y, datetimes)

    print(f"Test X shape: {X_test.shape}", flush=True)
    print(f"Test y shape: {y_test.shape}", flush=True)

    model = load_model()
    y_pred = np.clip(predict(model, X_test), 0, MAX_RUL_HOURS)

    metrics, matrix = calculate_day_metrics(y_test, y_pred)
    print_comparison_table(metrics)
    print_per_day_support(metrics)
    print_confusion_matrix(matrix)


if __name__ == "__main__":
    main()
