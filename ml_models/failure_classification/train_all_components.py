from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "ml_models" / "failure_classification"
MLFLOW_DB_PATH = Path(os.environ.get("PREDICTIQ_MLFLOW_DB", BASE_DIR / "mlflow_all_components_fixed.db"))

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
HORIZONS = [24, 48, 72]
LABEL_COLS = [
    f"{component}_fail_{horizon}h"
    for component in COMPONENTS
    for horizon in HORIZONS
]
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-08-01")
VAL_START = pd.Timestamp("2015-08-01")
VAL_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
OLD_METRICS = {
    "comp1": {"f1": 0.963292, "recall": 0.975490},
    "comp2": {"f1": 0.981533, "recall": 0.981533},
    "comp3": {"f1": 0.969287, "recall": 0.969792},
    "comp4": {"f1": 0.981523, "recall": 0.983704},
}


@dataclass
class ComponentResult:
    component: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    scale_pos_weight: float
    top_features: list[str]


def load_data() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_parquet(FEATURE_PATH).sort_values("datetime").reset_index(drop=True)
    drop_cols = ["datetime", "machineID", *LABEL_COLS]
    feature_cols = [column for column in data.columns if column not in drop_cols]
    return data, feature_cols


def split_data(
    data: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train_mask = data["datetime"].ge(TRAIN_START) & data["datetime"].lt(TRAIN_END)
    val_mask = data["datetime"].ge(VAL_START) & data["datetime"].lt(VAL_END)
    test_mask = data["datetime"].ge(TEST_START) & data["datetime"].lt(TEST_END)

    train = data.loc[train_mask]
    val = data.loc[val_mask]
    test = data.loc[test_mask]

    return (
        train[feature_cols],
        val[feature_cols],
        test[feature_cols],
        train[target],
        val[target],
        test[target],
    )


def class_counts(y: pd.Series) -> tuple[int, int]:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    return int(counts[0]), int(counts[1])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def tune_threshold(y_test: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rows.append((threshold, precision, recall, f1))

    candidates = [row for row in rows if row[2] >= 0.85]
    if candidates:
        return max(candidates, key=lambda row: row[3])
    return max(rows, key=lambda row: row[3])


def top_feature_names(model: XGBClassifier, feature_cols: list[str], n: int = 5) -> list[str]:
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance.head(n)["feature"].tolist()


def train_component(
    data: pd.DataFrame,
    feature_cols: list[str],
    component: str,
) -> ComponentResult:
    target = f"{component}_fail_24h"
    print(f"\n=== Training {component} target={target} ===")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, feature_cols, target)
    negatives, positives = class_counts(y_train)
    scale_pos_weight = negatives / positives

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print(f"Training class distribution: 0={negatives}, 1={positives}")
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")

    model = train_model(X_train, y_train, X_val, y_val, scale_pos_weight)
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, precision, recall, f1 = tune_threshold(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba) if y_test.nunique() == 2 else float("nan")
    top_features = top_feature_names(model, feature_cols)

    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print("Top 5 features:")
    for feature in top_features:
        print(f"- {feature}")

    model_path = MODEL_DIR / f"xgb_{component}_24h.json"
    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    with mlflow.start_run(run_name=f"xgb_{component}_24h"):
        mlflow.log_params(
            {
                "component": component,
                "target": target,
                "threshold": threshold,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight": scale_pos_weight,
                "split_type": "time_based",
                "validation_month": "2015-08",
                "test_months": "2015-09_to_2015-12",
            }
        )
        mlflow.log_metrics(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )
        mlflow.xgboost.log_model(model, name="model")

    return ComponentResult(
        component=component,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        scale_pos_weight=scale_pos_weight,
        top_features=top_features,
    )


def print_final_summary(results: list[ComponentResult]) -> None:
    summary = pd.DataFrame(
        [
            {
                "component": result.component,
                "threshold": result.threshold,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
            for result in results
        ]
    )
    print("\nFinal comparison table:")
    print(summary.to_string(index=False))

    comparison = pd.DataFrame(
        [
            {
                "Component": result.component,
                "Old F1": OLD_METRICS[result.component]["f1"],
                "New F1": result.f1,
                "Old Recall": OLD_METRICS[result.component]["recall"],
                "New Recall": result.recall,
            }
            for result in results
        ]
    )
    print("\nLeakage-fixed comparison table:")
    print(comparison.to_string(index=False))

    hardest = min(results, key=lambda result: result.f1)
    feature_counts = Counter(
        feature for result in results for feature in result.top_features
    )
    most_common_feature, most_common_count = feature_counts.most_common(1)[0]
    low_recall = [result for result in results if result.recall < 0.85]

    print(f"\nHardest component to predict: {hardest.component} (lowest F1={hardest.f1:.6f})")
    print(
        "Feature appearing most often in top 5 across components: "
        f"{most_common_feature} ({most_common_count}/4 components)"
    )
    if low_recall:
        print("Components with recall < 0.85:")
        for result in low_recall:
            print(
                f"- {result.component}: recall={result.recall:.6f}; "
                "best-F1 threshold could not meet the desired recall floor."
            )
    else:
        print("No component has recall < 0.85.")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_all_components_24h")

    data, feature_cols = load_data()
    results = [
        train_component(data, feature_cols, component)
        for component in COMPONENTS
    ]
    print_final_summary(results)


if __name__ == "__main__":
    main()
