from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
MODEL_PATH = MODEL_DIR / "rul_model_comp2_v4.pt"
PLOT_PATH = MODEL_DIR / "rul_prediction_plot_v4.png"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
COMPONENT = "comp2"
SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
ERROR_IDS = ["error1", "error2", "error3", "error4", "error5"]
FEATURE_COLS = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
    "volt_mean_6h",
    "rotate_mean_6h",
    "pressure_mean_6h",
    "vibration_mean_6h",
    "volt_std_6h",
    "rotate_std_6h",
    "pressure_std_6h",
    "vibration_std_6h",
    "error1_count_6h",
    "error2_count_6h",
    "error3_count_6h",
    "error4_count_6h",
    "error5_count_6h",
    "hours_since_last_error",
    "days_since_comp_maint",
    "days_since_comp2_maint",
    "age",
    "model_model1",
    "model_model2",
    "model_model3",
    "model_model4",
]
SEQUENCE_LENGTH = 48
MAX_RUL_HOURS = 100
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
EPOCHS = 100
BATCH_SIZE = 256
RANDOM_SEED = 42
BASE_LR = 0.001
WARMUP_LR = 0.0001
V2_METRICS = {"mae": 10.53, "r2": 0.7318, "accuracy_24h": 0.8941}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1)


class RULModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.attention = SelfAttention(hidden_size=128)
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.lstm(x)
        context = self.attention(lstm_output)
        return self.output(context).squeeze(-1)


def set_seed() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv", parse_dates=["datetime"])
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    machines = pd.read_csv(RAW_DIR / "PdM_machines.csv")
    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    maint = maint.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return telemetry, failures, maint, errors, machines


def add_maintenance_recency(
    telemetry_features: pd.DataFrame,
    maint: pd.DataFrame,
    component: str,
    output_col: str,
) -> pd.DataFrame:
    base = telemetry_features[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()
    comp_maint = (
        maint.loc[maint["comp"].eq(component), ["machineID", "datetime"]]
        .rename(columns={"datetime": "last_maint_datetime"})
        .sort_values(["last_maint_datetime", "machineID"])
    )
    merged = pd.merge_asof(
        base,
        comp_maint,
        left_on="datetime",
        right_on="last_maint_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    telemetry_features[output_col] = (
        (merged["datetime"] - merged["last_maint_datetime"]).dt.total_seconds() / 86400
    ).sort_index().to_numpy()
    telemetry_features[output_col] = telemetry_features[output_col].fillna(999.0)
    return telemetry_features


def enrich_telemetry(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    enriched = telemetry.copy()
    grouped = enriched.groupby("machineID", group_keys=False)
    for sensor in SENSOR_COLS:
        rolling = grouped[sensor].rolling(window=6, min_periods=6)
        enriched[f"{sensor}_mean_6h"] = rolling.mean().reset_index(level=0, drop=True)
        enriched[f"{sensor}_std_6h"] = rolling.std().reset_index(level=0, drop=True)

    error_counts = (
        pd.crosstab([errors["machineID"], errors["datetime"]], errors["errorID"])
        .reindex(columns=ERROR_IDS, fill_value=0)
        .reset_index()
    )
    enriched = enriched.merge(error_counts, on=["machineID", "datetime"], how="left")
    enriched[ERROR_IDS] = enriched[ERROR_IDS].fillna(0)
    grouped = enriched.groupby("machineID", group_keys=False)
    for error_id in ERROR_IDS:
        enriched[f"{error_id}_count_6h"] = (
            grouped[error_id]
            .rolling(window=6, min_periods=6)
            .sum()
            .reset_index(level=0, drop=True)
        )

    error_events = (
        errors[["machineID", "datetime"]]
        .rename(columns={"datetime": "last_error_datetime"})
        .sort_values(["last_error_datetime", "machineID"])
    )
    base = enriched[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()
    last_errors = pd.merge_asof(
        base,
        error_events,
        left_on="datetime",
        right_on="last_error_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    enriched["hours_since_last_error"] = (
        (last_errors["datetime"] - last_errors["last_error_datetime"]).dt.total_seconds() / 3600
    ).sort_index().to_numpy()
    enriched["hours_since_last_error"] = enriched["hours_since_last_error"].fillna(9999.0)

    enriched = add_maintenance_recency(enriched, maint, COMPONENT, "days_since_comp_maint")
    enriched = add_maintenance_recency(enriched, maint, "comp2", "days_since_comp2_maint")

    machines_encoded = pd.get_dummies(
        machines[["machineID", "model", "age"]],
        columns=["model"],
        prefix="model",
        dtype="int8",
    )
    for model_col in ["model_model1", "model_model2", "model_model3", "model_model4"]:
        if model_col not in machines_encoded.columns:
            machines_encoded[model_col] = 0
    enriched = enriched.merge(machines_encoded, on="machineID", how="left")

    return (
        enriched.drop(columns=ERROR_IDS)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[col for col in FEATURE_COLS if col != "hours_since_last_error"])
        .reset_index(drop=True)
    )


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
        machine_telemetry = telemetry.loc[telemetry["machineID"].eq(machine_id), ["datetime", "machineID"]]
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
    print("RUL row counts by component:")
    for component in COMPONENTS:
        table = build_component_rul_table(telemetry, failures, maint, component)
        tables[component] = table
        print(f"{component}: {len(table)}")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (datetimes >= np.datetime64(TRAIN_START)) & (datetimes < np.datetime64(TRAIN_END))
    test_mask = (datetimes >= np.datetime64(TEST_START)) & (datetimes < np.datetime64(TEST_END))
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
        y_train,
        y_test,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RULModel:
    pin_memory = DEVICE.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    model = RULModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.1 if epoch < 5 else 1.0,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

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
            optimizer.step()
            train_losses.append(loss.item() * len(batch_X))
        train_loss = sum(train_losses) / len(train_dataset)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item() * len(batch_X))
        val_loss = sum(val_losses) / len(test_dataset)

        if epoch <= 5:
            warmup_scheduler.step()
        elif epoch > 20:
            plateau_scheduler.step(val_loss)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch}, train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def predict(model: RULModel, X: np.ndarray) -> np.ndarray:
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
        "accuracy_6h": float(np.mean(absolute_error <= 6)),
        "accuracy_12h": float(np.mean(absolute_error <= 12)),
        "accuracy_24h": float(np.mean(absolute_error <= 24)),
    }


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    sample_size = min(1000, len(y_true))
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(y_true), size=sample_size, replace=False)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true[indices], y_pred[indices], alpha=0.35, s=12)
    plt.plot([0, MAX_RUL_HOURS], [0, MAX_RUL_HOURS], color="red", linewidth=2)
    plt.xlabel("Actual RUL hours")
    plt.ylabel("Predicted RUL hours")
    plt.title("Predicted vs Actual RUL - comp2 v4")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def print_comparison(metrics: dict[str, float]) -> None:
    comparison = pd.DataFrame(
        [
            {"Metric": "MAE", "V2": f"{V2_METRICS['mae']:.2f}h", "V4": f"{metrics['mae']:.2f}h"},
            {"Metric": "R2", "V2": f"{V2_METRICS['r2']:.4f}", "V4": f"{metrics['r2']:.4f}"},
            {
                "Metric": "Acc_24h",
                "V2": f"{V2_METRICS['accuracy_24h'] * 100:.2f}%",
                "V4": f"{metrics['accuracy_24h'] * 100:.2f}%",
            },
        ]
    )
    print("\nV2 / V4 comparison:")
    print(comparison.to_string(index=False))


def main() -> None:
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    telemetry, failures, maint, errors, machines = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors, maint, machines)
    rul_tables = build_all_rul_tables(telemetry, failures, maint)
    comp_rul = rul_tables[COMPONENT]

    X, y, datetimes = build_sequences(telemetry_features, comp_rul)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(
        "RUL distribution: "
        f"min={y.min():.2f}, max={y.max():.2f}, "
        f"mean={y.mean():.2f}, median={np.median(y):.2f}"
    )

    X_train, X_test, y_train, y_test = split_and_scale(X, y, datetimes)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    model = train_model(X_train, y_train, X_test, y_test)
    y_pred = np.clip(predict(model, X_test), 0, MAX_RUL_HOURS)
    metrics = evaluate(y_test, y_pred)

    print("\nTest metrics:")
    print(f"MAE: {metrics['mae']:.6f} hours")
    print(f"RMSE: {metrics['rmse']:.6f} hours")
    print(f"R2 Score: {metrics['r2']:.6f}")
    print(f"Accuracy within 6 hours: {metrics['accuracy_6h']:.6f}")
    print(f"Accuracy within 12 hours: {metrics['accuracy_12h']:.6f}")
    print(f"Accuracy within 24 hours: {metrics['accuracy_24h']:.6f}")
    print_comparison(metrics)

    save_prediction_plot(y_test, y_pred)
    torch.save(model.state_dict(), MODEL_PATH)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_rul_v4")
    with mlflow.start_run(run_name=f"rul_model_{COMPONENT}_v4"):
        mlflow.log_params(
            {
                "component": COMPONENT,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": BASE_LR,
                "warmup_learning_rate": WARMUP_LR,
                "sequence_length": SEQUENCE_LENGTH,
                "rul_cap_hours": MAX_RUL_HOURS,
                "feature_count": len(FEATURE_COLS),
                "architecture": "cnn_lstm_attention",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PLOT_PATH))

    print(f"Saved prediction plot to: {PLOT_PATH}")
    print(f"Saved RUL model to: {MODEL_PATH}")
    print("RUL model training complete.")


if __name__ == "__main__":
    main()
