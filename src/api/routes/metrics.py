"""Metrics API endpoints."""

from fastapi import APIRouter, status

from src.core.logging import get_logger
from src.models.response import MetricsResponse

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get metrics",
    description="Get aggregated statistics for all models and consensus.",
)
async def get_metrics() -> MetricsResponse:
    """Get aggregated metrics and statistics.

    Returns overall statistics including per-model accuracy,
    latency, and consensus rates.
    """
    logger.info("metrics_requested")

    # TODO: Implement in Phase 5
    # For now, return placeholder response

    return MetricsResponse(
        total_requests=0,
        models={},
        consensus={
            "unanimous_rate": 0.0,
            "majority_rate": 0.0,
            "disagreement_rate": 0.0,
        },
    )
