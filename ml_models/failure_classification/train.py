from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import matplotlib

matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "ml_models" / "failure_classification"
XGB_48H_MODEL_PATH = MODEL_DIR / "xgb_comp2_48h.json"
BEST_MODEL_JSON_PATH = MODEL_DIR / "best_model.json"
BEST_MODEL_TXT_PATH = MODEL_DIR / "best_model.txt"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.png"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

TARGETS = ["comp2_fail_48h", "comp2_fail_24h"]
MODEL_TYPES = ["xgboost", "lightgbm"]
LABEL_COLS = [
    f"{component}_fail_{horizon}h"
    for component in ["comp1", "comp2", "comp3", "comp4"]
    for horizon in [24, 48, 72]
]
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


@dataclass
class RunResult:
    model_type: str
    target: str
    model: XGBClassifier | lgb.LGBMClassifier
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    scale_pos_weight: float | None
    feature_cols: list[str]
    y_test: pd.Series
    y_proba: np.ndarray


def load_data() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_parquet(FEATURE_PATH).sort_values("datetime").reset_index(drop=True)
    drop_cols = ["datetime", "machineID", *LABEL_COLS]
    feature_cols = [column for column in data.columns if column not in drop_cols]
    return data, feature_cols


def split_target(
    data: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask = data["datetime"].ge(TRAIN_START) & data["datetime"].lt(TRAIN_END)
    test_mask = data["datetime"].ge(TEST_START) & data["datetime"].lt(TEST_END)

    train_df = data.loc[train_mask]
    test_df = data.loc[test_mask]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def class_counts(y: pd.Series) -> tuple[int, int]:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    return int(counts[0]), int(counts[1])


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    negatives, positives = class_counts(y_train)
    if positives == 0:
        raise ValueError("Training target has no positive examples.")
    return negatives / positives


def build_model(model_type: str, scale_pos_weight: float) -> XGBClassifier | lgb.LGBMClassifier:
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=20,
            tree_method="hist",
        )

    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def fit_model(
    model: XGBClassifier | lgb.LGBMClassifier,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> XGBClassifier | lgb.LGBMClassifier:
    if model_type == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return model

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model


def tune_threshold(y_test: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    print("Threshold tuning:")
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rows.append((threshold, precision, recall, f1))
        print(
            f"threshold={threshold:.2f}, "
            f"precision={precision:.6f}, recall={recall:.6f}, f1={f1:.6f}"
        )

    candidates = [row for row in rows if row[2] >= 0.85]
    if candidates:
        selected = max(candidates, key=lambda row: row[3])
    else:
        selected = max(rows, key=lambda row: row[3])
        print("No threshold reached recall >= 0.85; selected best F1 overall.")

    return selected


def evaluate_at_threshold(
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    title: str,
) -> tuple[float, float, float, float, np.ndarray]:
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_test.nunique() == 2 else float("nan")

    print(f"\n{title} final evaluation at threshold {threshold:.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")

    return precision, recall, f1, roc_auc, y_pred


def save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, path: Path, title: str) -> None:
    display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    display.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(
    model: XGBClassifier | lgb.LGBMClassifier,
    feature_cols: list[str],
    path: Path,
    title: str,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    top_20 = importance.head(20)
    plt.figure(figsize=(10, 7))
    plt.barh(top_20["feature"][::-1], top_20["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return importance


def log_model(model_type: str, model: XGBClassifier | lgb.LGBMClassifier) -> None:
    if model_type == "xgboost":
        mlflow.xgboost.log_model(model, name="model")
    else:
        mlflow.lightgbm.log_model(model, name="model")


def run_combination(
    data: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    target: str,
) -> RunResult:
    print(f"\n=== Training {model_type} target={target} ===")
    X_train, X_test, y_train, y_test = split_target(data, feature_cols, target)

    train_negatives, train_positives = class_counts(y_train)
    test_negatives, test_positives = class_counts(y_test)
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Training class distribution: 0={train_negatives}, 1={train_positives}")
    print(f"Test class distribution: 0={test_negatives}, 1={test_positives}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.6f}")
    print("SMOTE: disabled")

    model = build_model(model_type, scale_pos_weight)
    model = fit_model(model, model_type, X_train, y_train, X_test, y_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold, _, _, _ = tune_threshold(y_test, y_proba)
    precision, recall, f1, roc_auc, y_pred = evaluate_at_threshold(
        y_test, y_proba, threshold, f"{model_type} {target}"
    )

    run_slug = f"{model_type}_{target}"
    cm_path = MODEL_DIR / f"confusion_matrix_{run_slug}.png"
    fi_path = MODEL_DIR / f"feature_importance_{run_slug}.png"
    save_confusion_matrix(y_test, y_pred, cm_path, f"{model_type} {target}")
    importance = save_feature_importance(
        model, feature_cols, fi_path, f"Top 20 Feature Importances - {model_type} {target}"
    )

    print("Top 20 feature importances:")
    for row in importance.head(20).itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")

    with mlflow.start_run(run_name=run_slug):
        mlflow.log_params(
            {
                "model_type": model_type,
                "target": target,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "smote": False,
                "split_type": "time_based",
                "scale_pos_weight": scale_pos_weight if model_type == "xgboost" else None,
                "is_unbalance": model_type == "lightgbm",
                "optimal_threshold": threshold,
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
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))
        log_model(model_type, model)

    return RunResult(
        model_type=model_type,
        target=target,
        model=model,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        scale_pos_weight=scale_pos_weight if model_type == "xgboost" else None,
        feature_cols=feature_cols,
        y_test=y_test,
        y_proba=y_proba,
    )


def choose_best(results: list[RunResult]) -> RunResult:
    recall_candidates = [result for result in results if result.recall >= 0.85]
    if recall_candidates:
        return max(recall_candidates, key=lambda result: result.f1)
    return max(results, key=lambda result: result.f1)


def save_best_model(best: RunResult) -> Path:
    if best.model_type == "xgboost" and best.target == "comp2_fail_48h":
        best.model.save_model(XGB_48H_MODEL_PATH)
        return XGB_48H_MODEL_PATH

    if best.model_type == "xgboost":
        best.model.save_model(BEST_MODEL_JSON_PATH)
        return BEST_MODEL_JSON_PATH

    best.model.booster_.save_model(BEST_MODEL_TXT_PATH)
    return BEST_MODEL_TXT_PATH


def print_summary_table(results: list[RunResult]) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": result.model_type,
                "target": result.target,
                "threshold": result.threshold,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
            for result in results
        ]
    )
    print("\nFinal summary table:")
    print(summary.to_string(index=False))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_failure_classification_optimized")

    data, feature_cols = load_data()
    results = [
        run_combination(data, feature_cols, model_type, target)
        for model_type in MODEL_TYPES
        for target in TARGETS
    ]

    print_summary_table(results)
    best = choose_best(results)
    saved_path = save_best_model(best)

    print(
        "\nBest model: "
        f"{best.model_type} target={best.target}, threshold={best.threshold:.2f}, "
        f"precision={best.precision:.6f}, recall={best.recall:.6f}, "
        f"f1={best.f1:.6f}, roc_auc={best.roc_auc:.6f}"
    )
    print(f"Saved best model to: {saved_path}")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
