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
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml_models" / "rul_regression"
MODEL_PATH = MODEL_DIR / "rul_model_comp2_daily.pt"
MLFLOW_DB_PATH = Path(os.environ.get("PREDICTIQ_MLFLOW_DB", BASE_DIR / "mlflow_daily.db"))

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
MAX_RUL_HOURS = 120
NUM_CLASSES = 5
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class RULDailyClassifier(nn.Module):
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
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context, _ = self.attention(lstm_output)
        return self.output(context)


def set_seed() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def rul_to_label(rul_hours: pd.Series) -> pd.Series:
    return np.ceil(rul_hours / 24).clip(1, NUM_CLASSES).astype("int64") - 1


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


def build_component_daily_table(
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
            window["rul_hours"] = ((failure_time - window["datetime"]).dt.total_seconds() / 3600).astype(int)
            rows.append(
                window.loc[
                    window["rul_hours"].between(1, MAX_RUL_HOURS),
                    ["datetime", "machineID", "rul_hours"],
                ]
            )

    if not rows:
        return pd.DataFrame(columns=["datetime", "machineID", "rul_hours", "label"])

    table = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["machineID", "datetime", "rul_hours"])
        .drop_duplicates(["machineID", "datetime"], keep="first")
        .sort_values(["machineID", "datetime"])
        .reset_index(drop=True)
    )
    table["label"] = rul_to_label(table["rul_hours"])
    return table


def build_all_daily_tables(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables = {}
    print("Daily RUL row counts by component:", flush=True)
    for component in COMPONENTS:
        table = build_component_daily_table(telemetry, failures, maint, component)
        tables[component] = table
        print(f"{component}: {len(table)}", flush=True)
    return tables


def build_sequences(
    telemetry_features: pd.DataFrame,
    daily_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    telemetry_by_machine = {
        machine_id: group.sort_values("datetime").reset_index(drop=True)
        for machine_id, group in telemetry_features.groupby("machineID")
    }
    X_rows = []
    y_rows = []
    datetime_rows = []
    for row in daily_table.itertuples(index=False):
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
        y_rows.append(int(row.label))
        datetime_rows.append(row.datetime)

    return (
        np.stack(X_rows).astype(np.float32),
        np.array(y_rows, dtype=np.int64),
        np.array(datetime_rows, dtype="datetime64[ns]"),
    )


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    datetimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (datetimes >= np.datetime64(TRAIN_START)) & (datetimes < np.datetime64(TRAIN_END))
    val_mask = (datetimes >= np.datetime64(TRAIN_END)) & (datetimes < np.datetime64(VAL_END))
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
) -> RULDailyClassifier:
    pin_memory = DEVICE.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    model = RULDailyClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
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
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
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
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item() * len(batch_X))
        val_loss = sum(val_losses) / len(val_dataset)
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
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


def predict_classes(model: RULDailyClassifier, X: np.ndarray) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")
    preds = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            logits = model(batch_X)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | np.ndarray]:
    labels = np.arange(NUM_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    exact_accuracy = float(np.mean(y_pred == y_true))
    within_1_day_accuracy = float(np.mean(np.abs(y_pred - y_true) <= 1))
    per_class_accuracy = []
    for label in labels:
        mask = y_true == label
        per_class_accuracy.append(float(np.mean(y_pred[mask] == y_true[mask])) if mask.any() else 0.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "per_class_accuracy": np.array(per_class_accuracy),
        "exact_accuracy": exact_accuracy,
        "within_1_day_accuracy": within_1_day_accuracy,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix,
    }


def print_summary(metrics: dict[str, float | np.ndarray]) -> None:
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = metrics["support"]
    per_class_accuracy = metrics["per_class_accuracy"]
    matrix = metrics["confusion_matrix"]

    print("\nPer-class accuracy:")
    for index, accuracy in enumerate(per_class_accuracy):
        print(f"Day {index + 1} accuracy: {accuracy:.6f}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nFinal summary:")
    print("Day   | Support | Precision | Recall | F1")
    for index in range(NUM_CLASSES):
        print(
            f"Day {index + 1} | {int(support[index])}       | "
            f"{precision[index]:.6f}  | {recall[index]:.6f} | {f1[index]:.6f}"
        )

    print(f"\nWeighted F1: {metrics['weighted_f1']:.6f}")
    print(f"Day 1 recall / final-24h classifier recall proxy: {recall[0]:.6f}")
    print(f"Overall exact accuracy: {metrics['exact_accuracy']:.6f}")
    print(f"Overall within-1-day accuracy: {metrics['within_1_day_accuracy']:.6f}")


def log_mlflow(metrics: dict[str, float | np.ndarray]) -> None:
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    per_class_accuracy = metrics["per_class_accuracy"]
    matrix = metrics["confusion_matrix"]

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_rul_daily")
    with mlflow.start_run(run_name=f"rul_model_{COMPONENT}_daily"):
        mlflow.log_params(
            {
                "architecture": "cnn_bilstm_bahdanau_attention_daily_classifier",
                "component": COMPONENT,
                "num_classes": NUM_CLASSES,
                "sequence_length": SEQUENCE_LENGTH,
                "max_rul_hours": MAX_RUL_HOURS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": BASE_LR,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "loss": "CrossEntropyLoss",
                "hidden_size": HIDDEN_SIZE,
                "bidirectional": True,
                "gradient_clip_max_norm": 1.0,
            }
        )
        log_metrics = {
            "exact_accuracy": metrics["exact_accuracy"],
            "within_1_day_accuracy": metrics["within_1_day_accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "day1_final_24h_recall_proxy": recall[0],
        }
        for index in range(NUM_CLASSES):
            day = index + 1
            log_metrics[f"day{day}_accuracy"] = per_class_accuracy[index]
            log_metrics[f"day{day}_precision"] = precision[index]
            log_metrics[f"day{day}_recall"] = recall[index]
            log_metrics[f"day{day}_f1"] = f1[index]
        mlflow.log_metrics(log_metrics)
        for true_index in range(NUM_CLASSES):
            for pred_index in range(NUM_CLASSES):
                mlflow.log_metric(
                    f"confusion_true_day{true_index + 1}_pred_day{pred_index + 1}",
                    int(matrix[true_index, pred_index]),
                )


def main() -> None:
    print("Starting daily RUL classification training", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    telemetry, failures, maint, errors = load_raw_data()
    telemetry_features = enrich_telemetry(telemetry, errors)
    daily_tables = build_all_daily_tables(telemetry, failures, maint)
    comp_daily = daily_tables[COMPONENT]

    print("\nLabel distribution before sequence filtering:")
    for label, count in comp_daily["label"].value_counts().sort_index().items():
        print(f"Day {int(label) + 1}: {int(count)}")

    X, y, datetimes = build_sequences(telemetry_features, comp_daily)
    print(f"\nX shape: {X.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print("Label distribution after sequence filtering:")
    for label in range(NUM_CLASSES):
        print(f"Day {label + 1}: {int(np.sum(y == label))}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y, datetimes)
    print(f"Train size: {len(X_train)}", flush=True)
    print(f"Validation size: {len(X_val)}", flush=True)
    print(f"Test size: {len(X_test)}", flush=True)

    model = train_model(X_train, y_train, X_val, y_val)
    y_pred = predict_classes(model, X_test)
    metrics = evaluate(y_test, y_pred)
    print_summary(metrics)

    torch.save(model.state_dict(), MODEL_PATH)
    log_mlflow(metrics)

    print(f"\nSaved RUL daily model to: {MODEL_PATH}")
    print(f"MLflow DB: {MLFLOW_DB_PATH}")
    print("Daily RUL classification training complete.")


if __name__ == "__main__":
    main()
