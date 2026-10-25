from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from api.dependencies import CLASSIFIER_FILES, ModelContainer, RUL_MODEL_PATH, get_models
from api.models import APIResponse, HealthStatus

router = APIRouter()


@router.get(
    "/health",
    response_model=APIResponse[HealthStatus],
    summary="Health check",
    description="Returns API status, loaded models, and uptime. "
                "Used by monitoring tools and deployment health checks.",
)
async def health_check(
    models: ModelContainer = Depends(get_models),
) -> APIResponse[HealthStatus]:
    """Return current API health status."""

    return APIResponse.ok(
        HealthStatus(
            status="healthy",
            models_loaded=True,
            classifier_models=list(CLASSIFIER_FILES.keys()),   # ["comp1","comp2","comp3","comp4"]
            rul_model=RUL_MODEL_PATH,
            uptime_seconds=time.time() - models.started_at,
            version="1.0.0",
        )
    )