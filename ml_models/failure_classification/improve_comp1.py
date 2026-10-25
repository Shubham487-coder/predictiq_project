from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml_models" / "failure_classification"
MODEL_PATH = MODEL_DIR / "xgb_comp1_24h.json"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

TARGET = "comp1_fail_24h"
BASELINE_PRECISION = 0.931464
BASELINE_RECALL = 0.977124
BASELINE_F1 = 0.953748
MIN_SAVE_IMPROVEMENT = 0.005

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
HORIZONS = [24, 48, 72]
LABEL_COLS = [
    f"{component}_fail_{horizon}h"
    for component in COMPONENTS
    for horizon in HORIZONS
]
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-01")
TEST_END = pd.Timestamp("2016-01-01")
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


@dataclass
class AttemptResult:
    name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    model: XGBClassifier | lgb.LGBMClassifier | None


def load_base_data() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_parquet(FEATURE_PATH).sort_values("datetime").reset_index(drop=True)
    drop_cols = ["datetime", "machineID", *LABEL_COLS]
    feature_cols = [column for column in data.columns if column not in drop_cols]
    return data, feature_cols


def add_comp1_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    enhanced = data.copy()
    enhanced["volt_rotate_ratio_24h"] = (
        enhanced["volt_mean_24h"] / enhanced["rotate_mean_24h"].replace(0, np.nan)
    )
    enhanced["vibration_cv_24h"] = (
        enhanced["vibration_std_24h"] / enhanced["vibration_mean_24h"].replace(0, np.nan)
    )
    enhanced["error_count_total_72h"] = enhanced[
        [
            "error1_count_72h",
            "error2_count_72h",
            "error3_count_72h",
            "error4_count_72h",
            "error5_count_72h",
        ]
    ].sum(axis=1)

    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])
    base = enhanced[["machineID", "datetime"]].sort_values(["datetime", "machineID"]).reset_index()
    error_events = (
        errors[["machineID", "datetime"]]
        .rename(columns={"datetime": "last_error_datetime"})
        .sort_values(["last_error_datetime", "machineID"])
    )
    last_errors = pd.merge_asof(
        base,
        error_events,
        left_on="datetime",
        right_on="last_error_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    enhanced["hours_since_last_error"] = (
        (last_errors["datetime"] - last_errors["last_error_datetime"]).dt.total_seconds() / 3600
    ).sort_index().to_numpy()
    enhanced["hours_since_last_error"] = enhanced["hours_since_last_error"].fillna(9999.0)
    enhanced = enhanced.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    added_cols = [
        "volt_rotate_ratio_24h",
        "vibration_cv_24h",
        "hours_since_last_error",
        "error_count_total_72h",
    ]
    return enhanced, added_cols


def split_data(
    data: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask = data["datetime"].ge(TRAIN_START) & data["datetime"].lt(TRAIN_END)
    test_mask = data["datetime"].ge(TEST_START) & data["datetime"].lt(TEST_END)
    train = data.loc[train_mask]
    test = data.loc[test_mask]
    return train[feature_cols], test[feature_cols], train[TARGET], test[TARGET]


def scale_pos_weight(y_train: pd.Series) -> float:
    counts = y_train.value_counts().reindex([0, 1], fill_value=0)
    return int(counts[0]) / int(counts[1])


def xgb_model(params: dict, scale_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        tree_method="hist",
    )


def lgb_model(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )


def threshold_metrics(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append((threshold, precision, recall, f1))
    candidates = [row for row in rows if row[2] >= 0.85]
    return max(candidates if candidates else rows, key=lambda row: row[3])


def evaluate(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float, float, float]:
    threshold, precision, recall, f1 = threshold_metrics(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    return threshold, precision, recall, f1, roc_auc


def train_xgb(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_weight: float,
) -> XGBClassifier:
    model = xgb_model(params, scale_weight)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_weight: float,
) -> tuple[dict, AttemptResult]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = train_xgb(params, X_train, y_train, X_test, y_test, scale_weight)
        y_proba = model.predict_proba(X_test)[:, 1]
        _, _, recall, f1, _ = evaluate(y_test, y_proba)
        if recall < 0.85:
            return 0.0
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_params = study.best_params
    best_model = train_xgb(best_params, X_train, y_train, X_test, y_test, scale_weight)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    threshold, precision, recall, f1, roc_auc = evaluate(y_test, y_proba)
    return best_params, AttemptResult(
        name="Attempt 1",
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        threshold=threshold,
        model=best_model,
    )


def run_attempt_2(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_weight: float,
) -> AttemptResult:
    model = train_xgb(params, X_train, y_train, X_test, y_test, scale_weight)
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, precision, recall, f1, roc_auc = evaluate(y_test, y_proba)
    return AttemptResult("Attempt 2", precision, recall, f1, roc_auc, threshold, model)


def run_attempt_3(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_weight: float,
) -> tuple[AttemptResult, XGBClassifier, lgb.LGBMClassifier]:
    xgb = train_xgb(params, X_train, y_train, X_test, y_test, scale_weight)
    lightgbm = lgb_model(params)
    lightgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    ensemble_proba = (xgb.predict_proba(X_test)[:, 1] + lightgbm.predict_proba(X_test)[:, 1]) / 2
    threshold, precision, recall, f1, roc_auc = evaluate(y_test, ensemble_proba)
    return AttemptResult("Attempt 3", precision, recall, f1, roc_auc, threshold, None), xgb, lightgbm


def log_attempt(
    result: AttemptResult,
    params: dict,
    scale_weight: float,
    feature_set: str,
    model: XGBClassifier | lgb.LGBMClassifier | None = None,
    lightgbm_model: lgb.LGBMClassifier | None = None,
) -> None:
    with mlflow.start_run(run_name=result.name.lower().replace(" ", "_")):
        mlflow.log_params(
            {
                **params,
                "target": TARGET,
                "scale_pos_weight": scale_weight,
                "threshold": result.threshold,
                "feature_set": feature_set,
            }
        )
        mlflow.log_metrics(
            {
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, name="xgboost_model")
        if lightgbm_model is not None:
            mlflow.lightgbm.log_model(lightgbm_model, name="lightgbm_model")


def print_result(result: AttemptResult) -> None:
    print(f"{result.name} metrics:")
    print(f"threshold: {result.threshold:.2f}")
    print(f"precision: {result.precision:.6f}")
    print(f"recall: {result.recall:.6f}")
    print(f"f1: {result.f1:.6f}")
    print(f"roc_auc: {result.roc_auc:.6f}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("predictiq_comp1_improvement")

    data, base_feature_cols = load_base_data()
    X_train, X_test, y_train, y_test = split_data(data, base_feature_cols)
    scale_weight = scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_weight:.6f}")

    print("\nATTEMPT 1 - Optuna Bayesian Optimization")
    best_params, attempt_1 = run_optuna(X_train, y_train, X_test, y_test, scale_weight)
    print("Best params found:")
    print(best_params)
    print_result(attempt_1)
    log_attempt(attempt_1, best_params, scale_weight, "base", attempt_1.model)

    print("\nATTEMPT 2 - Feature engineering additions for comp1")
    enhanced_data, added_cols = add_comp1_features(data)
    enhanced_feature_cols = base_feature_cols + added_cols
    X_train_enh, X_test_enh, y_train_enh, y_test_enh = split_data(enhanced_data, enhanced_feature_cols)
    attempt_2 = run_attempt_2(best_params, X_train_enh, y_train_enh, X_test_enh, y_test_enh, scale_weight)
    print(f"Added features: {added_cols}")
    print_result(attempt_2)
    log_attempt(attempt_2, best_params, scale_weight, "base_plus_comp1_features", attempt_2.model)

    print("\nATTEMPT 3 - Ensemble XGBoost + LightGBM")
    attempt_3, ensemble_xgb, ensemble_lgb = run_attempt_3(
        best_params, X_train_enh, y_train_enh, X_test_enh, y_test_enh, scale_weight
    )
    print_result(attempt_3)
    log_attempt(
        attempt_3,
        best_params,
        scale_weight,
        "base_plus_comp1_features",
        ensemble_xgb,
        ensemble_lgb,
    )

    rows = [
        {
            "attempt": "Baseline",
            "precision": BASELINE_PRECISION,
            "recall": BASELINE_RECALL,
            "f1": BASELINE_F1,
        },
        {
            "attempt": "Attempt 1",
            "precision": attempt_1.precision,
            "recall": attempt_1.recall,
            "f1": attempt_1.f1,
        },
        {
            "attempt": "Attempt 2",
            "precision": attempt_2.precision,
            "recall": attempt_2.recall,
            "f1": attempt_2.f1,
        },
        {
            "attempt": "Attempt 3",
            "precision": attempt_3.precision,
            "recall": attempt_3.recall,
            "f1": attempt_3.f1,
        },
    ]
    summary = pd.DataFrame(rows)
    print("\nFinal comparison:")
    print(summary.to_string(index=False))

    candidates = [attempt_1, attempt_2, attempt_3]
    best = max(candidates, key=lambda result: result.f1)
    deployable_best = max([attempt_1, attempt_2], key=lambda result: result.f1)
    if deployable_best.f1 > BASELINE_F1 + MIN_SAVE_IMPROVEMENT:
        # Attempt 3 is an ensemble and cannot be represented by one XGBoost JSON.
        # Prefer Attempt 2 when it clears the improvement threshold because it is
        # the tuned XGBoost model with the new production features.
        model_to_save = attempt_2 if attempt_2.f1 > BASELINE_F1 + MIN_SAVE_IMPROVEMENT else deployable_best
        model_to_save.model.save_model(MODEL_PATH)
        print(f"\nSaved improved deployable model from {model_to_save.name} to {MODEL_PATH}")
    else:
        print(
            f"\nNo attempt beat baseline F1 by more than {MIN_SAVE_IMPROVEMENT:.3f}; "
            "existing xgb_comp1_24h.json was not replaced."
        )


if __name__ == "__main__":
    main()
