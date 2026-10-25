from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml_models" / "rul_regression"
MLFLOW_DB_PATH = BASE_DIR / "mlflow_rul_all_components.db"

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
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
MAX_DAY = 5
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-08-01")
VAL_START = pd.Timestamp("2015-08-01")
VAL_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
EPOCHS = 150
BATCH_SIZE = 256
RANDOM_SEED = 42
BASE_LR = 0.001
WARMUP_LR = 0.0001
EARLY_STOPPING_PATIENCE = 25
LIMITED_FAILURE_THRESHOLD = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ComponentResult:
    component: str
    failures: int
    train_seq: int
    mae: float
    rmse: float
    r2: float
    acc_24h: float
    within_1_day: float
    day1_acc: float
    day2_acc: float
    day3_acc: float
    day4_acc: float
    hidden_size: int


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class RULBiLSTMAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attention = BahdanauAttention(hidden_size=hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context, _ = self.attention(lstm_output)
        return self.output(context).squeeze(-1)


def set_seed() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


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
            grouped[sensor].transform(
                lambda series: series.shift(1).rolling(window=6, min_periods=6).mean()
            )
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
        enriched.groupby("machineID")["error_count"].transform(
            lambda series: series.shift(1).rolling(window=6, min_periods=6).sum()
        )
    )
    return enriched.drop(columns=["error_count"]).reset_index(drop=True)


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
            rows.append(window.loc[window["rul"].between(1, MAX_RUL_HOURS), ["datetime", "machineID", "rul"]])

    if not rows:
        return pd.DataFrame(columns=["datetime", "machineID", "rul"])
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["machineID", "datetime", "rul"])
        .drop_duplicates(["machineID", "datetime"], keep="first")
        .sort_values(["machineID", "datetime"])
        .reset_index(drop=True)
    )


def build_sequences(
    telemetry_features: pd.DataFrame,
    rul_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    telemetry_by_machine = {
        machine_id: group.sort_values("datetime").reset_index(drop=True)
        for machine_id, group in telemetry_features.groupby("machineID")
    }
    X_rows = []
    y_rows = []
    datetime_rows = []
    dropped_nan_rows = 0
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
        if sequence[FEATURE_COLS].isna().any().any():
            dropped_nan_rows += 1
            continue
        X_rows.append(sequence[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_rows.append(float(row.rul))
        datetime_rows.append(row.datetime)

    return (
        np.stack(X_rows).astype(np.float32),
        np.array(y_rows, dtype=np.float32),
        np.array(datetime_rows, dtype="datetime64[ns]"),
        dropped_nan_rows,
    )


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    datetimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (datetimes >= np.datetime64(TRAIN_START)) & (datetimes < np.datetime64(TRAIN_END))
    val_mask = (datetimes >= np.datetime64(VAL_START)) & (datetimes < np.datetime64(VAL_END))
    test_mask = (datetimes >= np.datetime64(TEST_START)) & (datetimes < np.datetime64(TEST_END))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0

    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
        y_train,
        y_val,
        y_test,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
) -> RULBiLSTMAttention:
    pin_memory = DEVICE.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    model = RULBiLSTMAttention(hidden_size=hidden_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            set_optimizer_lr(optimizer, WARMUP_LR)
        elif epoch == 6:
            set_optimizer_lr(optimizer, BASE_LR)

        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item() * len(batch_X))
        train_loss = sum(train_losses) / len(train_dataset)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item() * len(batch_X))
        val_loss = sum(val_losses) / len(val_dataset)
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch}, train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, lr={current_lr:.6f}",
                flush=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def predict(model: RULBiLSTMAttention, X: np.ndarray) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")
    preds = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            preds.append(model(batch_X).detach().cpu().numpy())
    return np.concatenate(preds)


def hours_to_days(hours: np.ndarray) -> np.ndarray:
    clipped_hours = np.clip(hours, 1, MAX_RUL_HOURS)
    return np.ceil(clipped_hours / 24).clip(1, MAX_DAY).astype(np.int64)


def daily_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_day = hours_to_days(y_true)
    predicted_day = hours_to_days(y_pred)
    metrics = {
        "within_1_day": float(np.mean(np.abs(predicted_day - true_day) <= 1)),
    }
    for day in range(1, MAX_DAY + 1):
        mask = true_day == day
        metrics[f"day{day}_acc"] = (
            float(np.mean(predicted_day[mask] == true_day[mask])) if mask.any() else float("nan")
        )
    return metrics


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    absolute_error = np.abs(y_pred - y_true)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc_24h": float(np.mean(absolute_error <= 24)),
    }
    metrics.update(daily_metrics(y_true, y_pred))
    return metrics


def print_component_distribution(component: str, y: np.ndarray) -> None:
    print(
        f"{component} RUL distribution: "
        f"min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}",
        flush=True,
    )


def train_component(
    component: str,
    telemetry: pd.DataFrame,
    telemetry_features: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
) -> ComponentResult:
    print(f"\n=== Training RUL model for {component} ===", flush=True)
    failure_count = int(failures["failure"].eq(component).sum())
    hidden_size = 64 if failure_count < LIMITED_FAILURE_THRESHOLD else 128
    if hidden_size == 64:
        print(f"Using smaller model for {component} (limited training data)", flush=True)

    rul_table = build_component_rul_table(telemetry, failures, maint, component)
    X, y, datetimes, dropped_nan_rows = build_sequences(telemetry_features, rul_table)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y, datetimes)

    print(f"Dropped sequences with NaN rolling features: {dropped_nan_rows}", flush=True)
    print(f"Total sequences: {len(X)}", flush=True)
    print(f"Train size: {len(X_train)}", flush=True)
    print(f"Validation size: {len(X_val)}", flush=True)
    print(f"Test size: {len(X_test)}", flush=True)
    print_component_distribution(component, y)

    model = train_model(X_train, y_train, X_val, y_val, hidden_size)
    y_pred = np.clip(predict(model, X_test), 0, MAX_RUL_HOURS)
    metrics = evaluate(y_test, y_pred)

    print("\nTest metrics:")
    print(f"MAE: {metrics['mae']:.6f} hours")
    print(f"RMSE: {metrics['rmse']:.6f} hours")
    print(f"R2 Score: {metrics['r2']:.6f}")
    print(f"Accuracy within 24 hours: {metrics['acc_24h']:.6f}")
    print(f"Within-1-day daily bucket accuracy: {metrics['within_1_day']:.6f}")
    print(f"Day 1 accuracy: {metrics['day1_acc']:.6f}")
    print(f"Day 2 accuracy: {metrics['day2_acc']:.6f}")
    print(f"Day 3 accuracy: {metrics['day3_acc']:.6f}")
    print(f"Day 4 accuracy: {metrics['day4_acc']:.6f}")

    model_path = MODEL_DIR / f"rul_model_{component}_bilstm_att.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved RUL model to: {model_path}", flush=True)

    with mlflow.start_run(run_name=f"{component}_bilstm_att"):
        mlflow.log_params(
            {
                "component": component,
                "hidden_size": hidden_size,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": BASE_LR,
                "sequence_length": SEQUENCE_LENGTH,
                "rul_cap": MAX_RUL_HOURS,
            }
        )
        mlflow.log_metrics(
            {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "acc_24h": metrics["acc_24h"],
                "within_1_day": metrics["within_1_day"],
                "day1_acc": metrics["day1_acc"],
                "day2_acc": metrics["day2_acc"],
                "day3_acc": metrics["day3_acc"],
                "day4_acc": metrics["day4_acc"],
            }
        )

    return ComponentResult(
        component=component,
        failures=failure_count,
        train_seq=len(X_train),
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        acc_24h=metrics["acc_24h"],
        within_1_day=metrics["within_1_day"],
        day1_acc=metrics["day1_acc"],
        day2_acc=metrics["day2_acc"],
        day3_acc=metrics["day3_acc"],
        day4_acc=metrics["day4_acc"],
        hidden_size=hidden_size,
    )


def percent(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def print_final_summary(results: list[ComponentResult]) -> None:
    print("\nComponent | Failures | Train_seq | MAE   | R2    | Acc_24h | Within-1-day | Day1  | Day2")
    print("----------|----------|-----------|-------|-------|---------|--------------|-------|-----")
    for result in results:
        print(
            f"{result.component:<9} | "
            f"{result.failures:<8} | "
            f"{result.train_seq:<9} | "
            f"{result.mae:<5.2f} | "
            f"{result.r2:<5.3f} | "
            f"{percent(result.acc_24h):<7} | "
            f"{percent(result.within_1_day):<12} | "
            f"{percent(result.day1_acc):<5} | "
            f"{percent(result.day2_acc)}"
        )

    hardest = min(results, key=lambda result: result.r2)
    easiest = max(results, key=lambda result: result.r2)
    average_mae = float(np.mean([result.mae for result in results]))
    average_within_1_day = float(np.mean([result.within_1_day for result in results]))

    print(f"\nHardest component to predict: {hardest.component} (lowest R2={hardest.r2:.6f})")
    print(f"Easiest component to predict: {easiest.component} (highest R2={easiest.r2:.6f})")
    print(f"Average MAE across all components: {average_mae:.6f} hours")
    print(f"Average within-1-day across all components: {average_within_1_day:.6f}")


def main() -> None:
    print("Starting all-component RUL BiLSTM-Attention training", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_rul_all_components")

    telemetry, failures, maint, errors = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors)

    results = [
        train_component(component, telemetry, telemetry_features, failures, maint)
        for component in COMPONENTS
    ]
    print_final_summary(results)
    print("All-component RUL training complete.")


if __name__ == "__main__":
    main()
