from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
import torch
from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_models
from api.models import APIResponse, BatchRULRequest, RULRequest, RULResult

logger = logging.getLogger(__name__)
router = APIRouter()

COMPONENT_INDEX = {"comp1": 0, "comp2": 1, "comp3": 2, "comp4": 3}


# ---------------------------------------------------------------------------
# Single inference (used by /rul)
# ---------------------------------------------------------------------------

def _run_rul(
    model: torch.nn.Module,
    sequence: np.ndarray,
    comp_idx: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> float:
    device = next(model.parameters()).device
    normalized = (sequence - norm_mean) / norm_std          # (48, 9)
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
    comp_tensor = torch.tensor([comp_idx], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(tensor, comp_tensor)
        rul_hours = float(outputs[comp_idx].squeeze())
        print(f"DEBUG input[:3]: {sequence[:3, 0]}, rul_hours={rul_hours}")
    return float(np.clip(rul_hours, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Vectorized batch inference (used by /rul/batch)
# ---------------------------------------------------------------------------

def _run_rul_batch(
    model: torch.nn.Module,
    sequences: list[np.ndarray],    # each (48, 9)
    comp_indices: list[int],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> list[float]:
    device = next(model.parameters()).device
    results = [0.0] * len(sequences)

    # Group by component index — model expects uniform comp_idx per batch
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comp_indices):
        groups[c].append(i)

    for comp_idx, indices in groups.items():
        batch = np.stack([(sequences[i] - norm_mean) / norm_std for i in indices])
        tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        comp_tensor = torch.tensor(
            [comp_idx] * len(indices), dtype=torch.long
        ).to(device)

        with torch.no_grad():
            outputs = model(tensor, comp_tensor)
            # outputs[comp_idx] has shape (N,) for this group
            preds = outputs[comp_idx].squeeze(-1)  # (N,)
            for local_i, global_i in enumerate(indices):
                results[global_i] = float(np.clip(float(preds[local_i]), 0.0, 100.0))

    return results


# ---------------------------------------------------------------------------
# Bucket helper — unchanged
# ---------------------------------------------------------------------------

def _hours_to_bucket(hours: float) -> tuple[int, str, str]:
    if hours <= 24.0:
        return 1, "Within 1 day", "CRITICAL"
    elif hours <= 48.0:
        return 2, "1-2 days", "URGENT"
    elif hours <= 72.0:
        return 3, "2-3 days", "MONITOR"
    elif hours <= 96.0:
        return 4, "3-4 days", "HEALTHY"
    else:
        return 5, "4+ days", "HEALTHY"


# ---------------------------------------------------------------------------
# POST /rul  — single prediction
# ---------------------------------------------------------------------------

async def _predict_rul_core(
    request: RULRequest,
    models: ModelContainer,
) -> APIResponse[RULResult]:
    """Plain async function — no Depends, callable directly."""
    sequence = np.array(request.sensor_sequence, dtype=np.float64)
    comp_idx = COMPONENT_INDEX[request.component]
    loop = asyncio.get_event_loop()

    rul_hours = await loop.run_in_executor(
        models.executor,
        _run_rul,
        models.rul_model,
        sequence,
        comp_idx,
        models.rul_norm_mean,
        models.rul_norm_std,
    )

    day_bucket, day_label, interpretation = _hours_to_bucket(rul_hours)
    result = RULResult(
        machine_id=request.machine_id,
        component=request.component,
        predicted_rul_hours=round(rul_hours, 2),
        day_bucket=day_bucket,
        day_label=day_label,
        interpretation=interpretation,
        timestamp=datetime.utcnow(),
    )
    return APIResponse.ok(result)


@router.post(
    "/rul",
    response_model=APIResponse[RULResult],
    summary="Remaining Useful Life prediction",
    description="Predict RUL in hours for one component of one machine. "
                "Requires 48 timesteps × 9 raw sensor features.",
)
async def predict_rul(
    request: RULRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[RULResult]:
    return await _predict_rul_core(request, models)


# ---------------------------------------------------------------------------
# POST /rul/batch  — vectorized batch prediction
# ---------------------------------------------------------------------------

@router.post(
    "/rul/batch",
    response_model=APIResponse[list[RULResult]],
    summary="Batch RUL prediction",
    description="Predict RUL for up to 100 machine components in one call. "
                "All sequences run in a single GPU forward pass.",
)
async def rul_batch(
    batch: BatchRULRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[list[RULResult]]:
    try:
        sequences = [
            np.array(req.sensor_sequence, dtype=np.float64)
            for req in batch.requests
        ]
        comp_indices = [COMPONENT_INDEX[req.component] for req in batch.requests]

        loop = asyncio.get_event_loop()
        rul_hours_list = await loop.run_in_executor(
            models.executor,
            _run_rul_batch,
            models.rul_model,
            sequences,
            comp_indices,
            models.rul_norm_mean,
            models.rul_norm_std,
        )

        results = []
        for req, rul_hours in zip(batch.requests, rul_hours_list):
            day_bucket, day_label, interpretation = _hours_to_bucket(rul_hours)
            results.append(RULResult(
                machine_id=req.machine_id,
                component=req.component,
                predicted_rul_hours=round(rul_hours, 2),
                day_bucket=day_bucket,
                day_label=day_label,
                interpretation=interpretation,
                timestamp=datetime.utcnow(),
            ))

        logger.info(f"rul_batch completed {len(results)} predictions")
        return APIResponse.ok(results)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return APIResponse.failure(str(exc))