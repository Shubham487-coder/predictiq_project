from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_models
from api.models import (
    APIResponse,
    BatchClassifierRequest,
    ClassifierRequest,
    ClassifierResult,
    ComponentPrediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()


FEATURE_NAMES = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_24h", "volt_std_24h", "rotate_mean_24h", "rotate_std_24h",
    "pressure_mean_24h", "pressure_std_24h", "vibration_mean_24h", "vibration_std_24h",
    "volt_mean_48h", "volt_std_48h", "rotate_mean_48h", "rotate_std_48h",
    "pressure_mean_48h", "pressure_std_48h", "vibration_mean_48h", "vibration_std_48h",
    "volt_mean_72h", "volt_std_72h", "rotate_mean_72h", "rotate_std_72h",
    "pressure_mean_72h", "pressure_std_72h", "vibration_mean_72h", "vibration_std_72h",
    "error1_count_24h", "error2_count_24h", "error3_count_24h", "error4_count_24h", "error5_count_24h",
    "error1_count_48h", "error2_count_48h", "error3_count_48h", "error4_count_48h", "error5_count_48h",
    "error1_count_72h", "error2_count_72h", "error3_count_72h", "error4_count_72h", "error5_count_72h",
    "days_since_comp1_maint", "days_since_comp2_maint", "days_since_comp3_maint", "days_since_comp4_maint",
    "age",
    "model_model1", "model_model2", "model_model3", "model_model4",
    "volt_rotate_ratio_24h", "vibration_cv_24h", "error_count_total_72h", "hours_since_last_error",
]


def _run_classifier(booster: xgb.Booster, features: list[float]) -> float:
    """Single-row inference — used by /classify."""
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    dmatrix = xgb.DMatrix(df)
    return float(booster.predict(dmatrix)[0])


def _run_classifier_batch(
    comp1: xgb.Booster,
    comp2: xgb.Booster,
    comp3: xgb.Booster,
    comp4: xgb.Booster,
    feature_matrix: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-row inference — used by /classify/batch.
    Builds ONE DMatrix for all rows, runs all 4 models once each.
    100x faster than calling _run_classifier per row.
    """
    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    dmatrix = xgb.DMatrix(df)
    return (
        comp1.predict(dmatrix),
        comp2.predict(dmatrix),
        comp3.predict(dmatrix),
        comp4.predict(dmatrix),
    )


def _make_component_prediction(prob: float, threshold: float) -> ComponentPrediction:
    alert = prob >= 0.5
    if prob >= 0.7:
        status = "CRITICAL"
    elif prob >= 0.5:
        status = "WARNING"
    else:
        status = "HEALTHY"
    return ComponentPrediction(failure_probability=prob, alert=alert, status=status, threshold_used=threshold)


def _compute_overall_status(
    c1: ComponentPrediction,
    c2: ComponentPrediction,
    c3: ComponentPrediction,
    c4: ComponentPrediction,
) -> tuple[str, int]:
    statuses = [c1.status, c2.status, c3.status, c4.status]
    critical_count = statuses.count("CRITICAL")
    warning_count  = statuses.count("WARNING")
    if critical_count > 0:
        return "CRITICAL", 1
    elif warning_count >= 2:
        return "WARNING", 2
    elif warning_count == 1:
        return "WARNING", 3
    else:
        return "HEALTHY", 4


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

@router.post(
    "/classify",
    response_model=APIResponse[ClassifierResult],
    summary="Single machine failure classification",
    description="Run all 4 XGBoost component classifiers for one machine. "
                "Returns failure probability per component with alert flags.",
)
async def classify_machine(
    request: ClassifierRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[ClassifierResult]:
    features = request.to_feature_array()
    loop = asyncio.get_event_loop()
    threshold = request.alert_threshold

    prob1, prob2, prob3, prob4 = await asyncio.gather(
        loop.run_in_executor(models.executor, _run_classifier, models.comp1, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp2, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp3, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp4, features),
    )

    c1 = _make_component_prediction(prob1, threshold)
    c2 = _make_component_prediction(prob2, threshold)
    c3 = _make_component_prediction(prob3, threshold)
    c4 = _make_component_prediction(prob4, threshold)
    overall_status, priority = _compute_overall_status(c1, c2, c3, c4)

    logger.info(
        f"classify machine_id={request.machine_id} ! "
        f"{overall_status} (priority {priority}) "
        f"probs=[{prob1:.3f}, {prob2:.3f}, {prob3:.3f}, {prob4:.3f}]"
    )

    result = ClassifierResult(
        machine_id=request.machine_id,
        comp1=c1, comp2=c2, comp3=c3, comp4=c4,
        overall_status=overall_status,
        priority=priority,
        timestamp=datetime.utcnow(),
    )
    return APIResponse.ok(result)


# ---------------------------------------------------------------------------
# POST /classify/batch
# ---------------------------------------------------------------------------

@router.post(
    "/classify/batch",
    response_model=APIResponse[list[ClassifierResult]],
    summary="Batch machine failure classification",
    description="Classify up to 100 machines in one call using vectorised XGBoost inference.",
)
async def classify_batch(
    batch: BatchClassifierRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[list[ClassifierResult]]:
    feature_matrix = [req.to_feature_array() for req in batch.requests]
    thresholds     = [req.alert_threshold for req in batch.requests]
    machine_ids    = [req.machine_id for req in batch.requests]

    loop = asyncio.get_event_loop()
    probs1, probs2, probs3, probs4 = await loop.run_in_executor(
        models.executor,
        _run_classifier_batch,
        models.comp1, models.comp2, models.comp3, models.comp4,
        feature_matrix,
    )

    results = []
    for i, (p1, p2, p3, p4) in enumerate(zip(probs1, probs2, probs3, probs4)):
        t = thresholds[i]
        c1 = _make_component_prediction(float(p1), t)
        c2 = _make_component_prediction(float(p2), t)
        c3 = _make_component_prediction(float(p3), t)
        c4 = _make_component_prediction(float(p4), t)
        overall_status, priority = _compute_overall_status(c1, c2, c3, c4)
        results.append(ClassifierResult(
            machine_id=machine_ids[i],
            comp1=c1, comp2=c2, comp3=c3, comp4=c4,
            overall_status=overall_status,
            priority=priority,
            timestamp=datetime.utcnow(),
        ))

    logger.info(f"classify_batch {len(results)} machines complete")
    return APIResponse.ok(results)