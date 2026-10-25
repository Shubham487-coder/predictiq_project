from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_models
from api.endpoints.classifier import classify_machine
from api.models import (
    APIResponse,
    ClassifierRequest,
    ClassifierResult,
    FleetPriorityList,
    MachineStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper   convert ClassifierResult to MachineStatus
# ---------------------------------------------------------------------------

def _to_machine_status(result: ClassifierResult) -> MachineStatus:
    """Flatten ClassifierResult into MachineStatus for fleet view."""
    return MachineStatus(
        machine_id=result.machine_id,
        overall_status=result.overall_status,
        priority=result.priority,
        components={
            "comp1": result.comp1,
            "comp2": result.comp2,
            "comp3": result.comp3,
            "comp4": result.comp4,
        },
        rul_estimates={},   # RUL not run in fleet scan   too expensive for 100 machines
        timestamp=result.timestamp,
    )


def _sort_key(status: MachineStatus) -> tuple[int, float]:
    """Sort by priority first, then by highest component failure probability."""
    max_prob = max(
        status.components[c].failure_probability
        for c in status.components
    )
    return (status.priority, -max_prob)   # priority ascending, prob descending


# ---------------------------------------------------------------------------
# POST /fleet/scan
# ---------------------------------------------------------------------------

@router.post(
    "/fleet/scan",
    response_model=APIResponse[FleetPriorityList],
    summary="Fleet-wide priority scan",
    description="Run failure classifier on all submitted machines concurrently. "
                "Returns a ranked priority list   most critical first.",
)
async def fleet_scan(
    requests: list[ClassifierRequest],
    models: ModelContainer = Depends(get_models),
) -> APIResponse[FleetPriorityList]:
    """Classify all machines and return ranked priority list."""

    if not requests:
        return APIResponse.failure("No machines submitted for fleet scan.")

    scan_start = time.perf_counter()

    # Run all classifiers concurrently
    responses = await asyncio.gather(
        *[classify_machine(req, models) for req in requests]
    )

    # Extract results from APIResponse envelopes
    results: list[ClassifierResult] = [r.data for r in responses]

    # Convert to MachineStatus and sort
    statuses = [_to_machine_status(r) for r in results]
    statuses.sort(key=_sort_key)

    # Count by priority bucket
    critical_count = sum(1 for s in statuses if s.priority == 1)
    urgent_count   = sum(1 for s in statuses if s.priority == 2)
    monitor_count  = sum(1 for s in statuses if s.priority == 3)
    healthy_count  = sum(1 for s in statuses if s.priority == 4)

    scan_duration_ms = (time.perf_counter() - scan_start) * 1000

    logger.info(
        f"fleet_scan {len(requests)} machines in {scan_duration_ms:.1f}ms   "
        f"critical={critical_count} urgent={urgent_count} "
        f"monitor={monitor_count} healthy={healthy_count}"
    )

    fleet = FleetPriorityList(
        total_machines=len(statuses),
        critical_count=critical_count,
        urgent_count=urgent_count,
        monitor_count=monitor_count,
        healthy_count=healthy_count,
        priority_list=statuses,
        scan_duration_ms=round(scan_duration_ms, 2),
        timestamp=datetime.utcnow(),
    )

    return APIResponse.ok(fleet)