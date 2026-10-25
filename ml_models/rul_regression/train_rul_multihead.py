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
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml_models" / "rul_regression"
MODEL_PATH = MODEL_DIR / "rul_model_multihead.pt"
MLFLOW_DB_PATH = BASE_DIR / "mlflow_rul_multihead.db"

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
COMPONENT_TO_INDEX = {component: index for index, component in enumerate(COMPONENTS)}
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
HIDDEN_SIZE = 128
MASK_VALUE = -1.0
SEPARATE_MODEL_METRICS = {
    "comp1": {"mae": 10.11, "r2": 0.761, "within_1_day": 0.9696},
    "comp2": {"mae": 9.55, "r2": 0.773, "within_1_day": 0.9677},
    "comp3": {"mae": 8.70, "r2": 0.802, "within_1_day": 0.9832},
    "comp4": {"mae": 8.71, "r2": 0.803, "within_1_day": 0.9822},
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


class RULMultiHead(nn.Module):
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
        self.attention = BahdanauAttention(HIDDEN_SIZE)
        self.head_comp1 = self._build_head()
        self.head_comp2 = self._build_head()
        self.head_comp3 = self._build_head()
        self.head_comp4 = self._build_head()

    @staticmethod
    def _build_head() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context, _ = self.attention(lstm_output)
        return (
            self.head_comp1(context),
            self.head_comp2(context),
            self.head_comp3(context),
            self.head_comp4(context),
        )


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


def build_component_sequences(
    telemetry_features: pd.DataFrame,
    rul_table: pd.DataFrame,
    component: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    telemetry_by_machine = {
        machine_id: group.sort_values("datetime").reset_index(drop=True)
        for machine_id, group in telemetry_features.groupby("machineID")
    }
    X_rows = []
    y_rows = []
    datetime_rows = []
    component_rows = []
    component_index = COMPONENT_TO_INDEX[component]

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
            continue

        target = np.full(len(COMPONENTS), MASK_VALUE, dtype=np.float32)
        target[component_index] = float(row.rul)
        X_rows.append(sequence[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_rows.append(target)
        datetime_rows.append(row.datetime)
        component_rows.append(component_index)

    return (
        np.stack(X_rows).astype(np.float32),
        np.stack(y_rows).astype(np.float32),
        np.array(datetime_rows, dtype="datetime64[ns]"),
        np.array(component_rows, dtype=np.int64),
    )


def build_combined_dataset(
    telemetry: pd.DataFrame,
    telemetry_features: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_parts = []
    y_parts = []
    datetime_parts = []
    component_parts = []

    print("Sequence counts by component:", flush=True)
    for component in COMPONENTS:
        rul_table = build_component_rul_table(telemetry, failures, maint, component)
        X_component, y_component, datetimes_component, components_component = build_component_sequences(
            telemetry_features,
            rul_table,
            component,
        )
        X_parts.append(X_component)
        y_parts.append(y_component)
        datetime_parts.append(datetimes_component)
        component_parts.append(components_component)
        print(f"{component}: {len(X_component)}", flush=True)

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    datetimes = np.concatenate(datetime_parts)
    component_labels = np.concatenate(component_parts)
    order = np.argsort(datetimes)
    print(f"Combined total: {len(X)}", flush=True)
    return X[order], y[order], datetimes[order], component_labels[order]


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    datetimes: np.ndarray,
    component_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (datetimes >= np.datetime64(TRAIN_START)) & (datetimes < np.datetime64(TRAIN_END))
    val_mask = (datetimes >= np.datetime64(VAL_START)) & (datetimes < np.datetime64(VAL_END))
    test_mask = (datetimes >= np.datetime64(TEST_START)) & (datetimes < np.datetime64(TEST_END))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_components = component_labels[test_mask]

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
        datetimes[train_mask],
        datetimes[val_mask],
        test_components,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def masked_mse_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    total_loss = 0.0
    count = 0
    for pred, target in zip(predictions, targets.T):
        mask = target != MASK_VALUE
        if mask.sum() > 0:
            total_loss = total_loss + F.mse_loss(pred[mask].squeeze(), target[mask])
            count += 1
    return total_loss / count


def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> RULMultiHead:
    pin_memory = DEVICE.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    model = RULMultiHead().to(DEVICE)
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
            predictions = model(batch_X)
            loss = masked_mse_loss(predictions, batch_y)
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
                predictions = model(batch_X)
                loss = masked_mse_loss(predictions, batch_y)
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


def predict(model: RULMultiHead, X: np.ndarray) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            predictions = model(batch_X)
            outputs.append(torch.cat(predictions, dim=1).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def hours_to_days(hours: np.ndarray) -> np.ndarray:
    clipped_hours = np.clip(hours, 1, MAX_RUL_HOURS)
    return np.ceil(clipped_hours / 24).clip(1, MAX_DAY).astype(np.int64)


def evaluate_component(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_components: np.ndarray,
    component: str,
) -> dict[str, float]:
    component_index = COMPONENT_TO_INDEX[component]
    component_mask = test_components == component_index
    y_true_component = y_test[component_mask, component_index]
    y_pred_component = y_pred[component_mask, component_index]

    mae = mean_absolute_error(y_true_component, y_pred_component)
    rmse = float(np.sqrt(mean_squared_error(y_true_component, y_pred_component)))
    r2 = r2_score(y_true_component, y_pred_component)
    absolute_error = np.abs(y_pred_component - y_true_component)

    true_day = hours_to_days(y_true_component)
    predicted_day = hours_to_days(y_pred_component)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc_24h": float(np.mean(absolute_error <= 24)),
        "within_1_day": float(np.mean(np.abs(predicted_day - true_day) <= 1)),
    }
    for day in range(1, MAX_DAY + 1):
        day_mask = true_day == day
        metrics[f"day{day}_acc"] = (
            float(np.mean(predicted_day[day_mask] == true_day[day_mask])) if day_mask.any() else float("nan")
        )
    return metrics


def evaluate_all_components(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_components: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        component: evaluate_component(y_test, y_pred, test_components, component)
        for component in COMPONENTS
    }


def percent(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def print_comparison_table(metrics_by_component: dict[str, dict[str, float]]) -> None:
    print("\nMulti-head model vs Separate models:")
    print("Component | Sep_MAE | MH_MAE | Sep_R2 | MH_R2 | Sep_W1D | MH_W1D")
    for component in COMPONENTS:
        sep = SEPARATE_MODEL_METRICS[component]
        mh = metrics_by_component[component]
        print(
            f"{component:<9} | "
            f"{sep['mae']:.2f}h  | "
            f"{mh['mae']:.2f}h | "
            f"{sep['r2']:.3f}  | "
            f"{mh['r2']:.3f} | "
            f"{percent(sep['within_1_day']):<7} | "
            f"{percent(mh['within_1_day'])}"
        )

    avg_sep_mae = float(np.mean([metrics["mae"] for metrics in SEPARATE_MODEL_METRICS.values()]))
    avg_sep_r2 = float(np.mean([metrics["r2"] for metrics in SEPARATE_MODEL_METRICS.values()]))
    avg_sep_w1d = float(np.mean([metrics["within_1_day"] for metrics in SEPARATE_MODEL_METRICS.values()]))
    avg_mh_mae = float(np.mean([metrics["mae"] for metrics in metrics_by_component.values()]))
    avg_mh_r2 = float(np.mean([metrics["r2"] for metrics in metrics_by_component.values()]))
    avg_mh_w1d = float(np.mean([metrics["within_1_day"] for metrics in metrics_by_component.values()]))
    print(
        f"{'Average':<9} | "
        f"{avg_sep_mae:.2f}h  | "
        f"{avg_mh_mae:.2f}h | "
        f"{avg_sep_r2:.3f}  | "
        f"{avg_mh_r2:.3f} | "
        f"{percent(avg_sep_w1d):<7} | "
        f"{percent(avg_mh_w1d)}"
    )


def log_mlflow(metrics_by_component: dict[str, dict[str, float]]) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_rul_multihead")
    with mlflow.start_run(run_name="multihead_bilstm_attention"):
        mlflow.log_params(
            {
                "architecture": "multihead_bilstm_attention",
                "shared_hidden": HIDDEN_SIZE,
                "num_heads": len(COMPONENTS),
                "sequence_length": SEQUENCE_LENGTH,
                "rul_cap": MAX_RUL_HOURS,
            }
        )
        metrics = {}
        for component in COMPONENTS:
            metrics[f"mae_{component}"] = metrics_by_component[component]["mae"]
            metrics[f"r2_{component}"] = metrics_by_component[component]["r2"]
            metrics[f"within1day_{component}"] = metrics_by_component[component]["within_1_day"]
        metrics["avg_mae"] = float(np.mean([value["mae"] for value in metrics_by_component.values()]))
        metrics["avg_r2"] = float(np.mean([value["r2"] for value in metrics_by_component.values()]))
        metrics["avg_within1day"] = float(
            np.mean([value["within_1_day"] for value in metrics_by_component.values()])
        )
        mlflow.log_metrics(metrics)


def save_model(model: RULMultiHead) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "components": COMPONENTS,
            "hidden_size": HIDDEN_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "rul_cap": MAX_RUL_HOURS,
            "feature_names": FEATURE_COLS,
        },
        MODEL_PATH,
    )
    print(f"Saved multi-head RUL model to: {MODEL_PATH}", flush=True)


def main() -> None:
    print("Starting multi-head RUL BiLSTM-Attention training", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    telemetry, failures, maint, errors = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors)
    X, y, datetimes, component_labels = build_combined_dataset(
        telemetry,
        telemetry_features,
        failures,
        maint,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, test_components = split_and_scale(
        X,
        y,
        datetimes,
        component_labels,
    )
    print(f"Train size: {len(X_train)}", flush=True)
    print(f"Validation size: {len(X_val)}", flush=True)
    print(f"Test size: {len(X_test)}", flush=True)

    model = train_model(X_train, y_train, X_val, y_val)
    y_pred = np.clip(predict(model, X_test), 0, MAX_RUL_HOURS)
    metrics_by_component = evaluate_all_components(y_test, y_pred, test_components)
    print_comparison_table(metrics_by_component)
    save_model(model)
    log_mlflow(metrics_by_component)
    print("Multi-head RUL training complete.")


if __name__ == "__main__":
    main()
