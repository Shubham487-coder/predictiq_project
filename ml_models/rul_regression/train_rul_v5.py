from __future__ import annotations

import os
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
MODEL_PATH = MODEL_DIR / "rul_model_comp2_v5.pt"
MLFLOW_DB_PATH = Path(os.environ.get("PREDICTIQ_MLFLOW_DB", BASE_DIR / "mlflow_v5.db"))

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
TRAIN_END = pd.Timestamp("2015-08-01")
VAL_START = pd.Timestamp("2015-08-01")
VAL_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
EPOCHS = 100
BATCH_SIZE = 256
RANDOM_SEED = 42
BASE_LR = 0.001
HIDDEN_SIZE = 128
V2_METRICS = {
    "mae": 10.53,
    "rmse": 14.97,
    "r2": 0.7318,
    "acc_6h": 0.4725,
    "acc_12h": 0.6421,
    "acc_24h": 0.8941,
}
BILSTM_ATT_METRICS = {
    "mae": 9.35,
    "rmse": 13.98,
    "r2": 0.7663,
    "acc_6h": 0.5273,
    "acc_12h": 0.6790,
    "acc_24h": 0.9164,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class ResidualOutputHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        residual = x
        x = self.residual(x)
        x = x + residual
        return self.output(x).squeeze(-1)


class RULV5Model(nn.Module):
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
        self.head = ResidualOutputHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context, _ = self.attention(lstm_output)
        return self.head(context)


def relative_mae_loss(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    denominator = torch.clamp(torch.abs(actual), min=1e-8)
    return torch.mean(torch.abs(predicted - actual) / denominator)


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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RULV5Model:
    pin_memory = DEVICE.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    model = RULV5Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = relative_mae_loss(preds, batch_y)
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
                loss = relative_mae_loss(preds, batch_y)
                val_losses.append(loss.item() * len(batch_X))
        val_loss = sum(val_losses) / len(val_dataset)
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0:
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
        if epochs_without_improvement >= 15:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def predict(model: RULV5Model, X: np.ndarray) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")
    preds = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            preds.append(model(batch_X).detach().cpu().numpy())
    return np.concatenate(preds)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    absolute_error = np.abs(y_pred - y_true)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc_6h": float(np.mean(absolute_error <= 6)),
        "acc_12h": float(np.mean(absolute_error <= 12)),
        "acc_24h": float(np.mean(absolute_error <= 24)),
    }


def print_comparison(metrics: dict[str, float]) -> None:
    print("\nMetric  | V2     | BiLSTM-Att | V5")
    print(f"MAE     | {V2_METRICS['mae']:.2f}h | {BILSTM_ATT_METRICS['mae']:.2f}h      | {metrics['mae']:.2f}h")
    print(f"RMSE    | {V2_METRICS['rmse']:.2f}h | {BILSTM_ATT_METRICS['rmse']:.2f}h     | {metrics['rmse']:.2f}h")
    print(f"R2      | {V2_METRICS['r2']:.4f} | {BILSTM_ATT_METRICS['r2']:.4f}     | {metrics['r2']:.4f}")
    print(f"Acc_6h  | {V2_METRICS['acc_6h'] * 100:.2f}% | {BILSTM_ATT_METRICS['acc_6h'] * 100:.2f}%     | {metrics['acc_6h'] * 100:.2f}%")
    print(f"Acc_12h | {V2_METRICS['acc_12h'] * 100:.2f}% | {BILSTM_ATT_METRICS['acc_12h'] * 100:.2f}%     | {metrics['acc_12h'] * 100:.2f}%")
    print(f"Acc_24h | {V2_METRICS['acc_24h'] * 100:.2f}% | {BILSTM_ATT_METRICS['acc_24h'] * 100:.2f}%     | {metrics['acc_24h'] * 100:.2f}%")


def log_mlflow(metrics: dict[str, float]) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_rul_v5")
    with mlflow.start_run(run_name=f"rul_model_{COMPONENT}_v5"):
        mlflow.log_params(
            {
                "architecture": "cnn_bilstm_bahdanau_attention_residual_head",
                "loss": "relative_mae",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": BASE_LR,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "bidirectional": True,
                "component": COMPONENT,
                "rul_cap_hours": MAX_RUL_HOURS,
                "train_end": str(TRAIN_END),
                "val_start": str(VAL_START),
                "val_end": str(VAL_END),
                "test_start": str(TEST_START),
                "weight_decay": 1e-4,
                "gradient_clip_max_norm": 1.0,
            }
        )
        mlflow.log_metrics(metrics)


def main() -> None:
    print("Starting RUL V5 training", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    telemetry, failures, maint, errors = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors)
    rul_tables = build_all_rul_tables(telemetry, failures, maint)
    comp_rul = rul_tables[COMPONENT]

    X, y, datetimes = build_sequences(telemetry_features, comp_rul)
    print(f"X shape: {X.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print(
        "RUL distribution: "
        f"min={y.min():.2f}, max={y.max():.2f}, "
        f"mean={y.mean():.2f}, median={np.median(y):.2f}",
        flush=True,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y, datetimes)
    print(f"Train size: {len(X_train)}", flush=True)
    print(f"Validation size: {len(X_val)}", flush=True)
    print(f"Test size: {len(X_test)}", flush=True)

    model = train_model(X_train, y_train, X_val, y_val)
    y_pred = np.clip(predict(model, X_test), 0, MAX_RUL_HOURS)
    metrics = evaluate(y_test, y_pred)

    print("\nTest metrics:")
    print(f"MAE: {metrics['mae']:.6f} hours")
    print(f"RMSE: {metrics['rmse']:.6f} hours")
    print(f"R2 Score: {metrics['r2']:.6f}")
    print(f"Accuracy within 6 hours: {metrics['acc_6h']:.6f}")
    print(f"Accuracy within 12 hours: {metrics['acc_12h']:.6f}")
    print(f"Accuracy within 24 hours: {metrics['acc_24h']:.6f}")
    print_comparison(metrics)

    torch.save(model.state_dict(), MODEL_PATH)
    log_mlflow(metrics)

    print(f"Saved RUL model to: {MODEL_PATH}")
    print(f"MLflow DB: {MLFLOW_DB_PATH}")
    print("RUL V5 training complete.")


if __name__ == "__main__":
    main()
