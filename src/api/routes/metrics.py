"""Metrics API endpoints."""

from fastapi import APIRouter, Depends, status

from src.api.dependencies import get_benchmark_repository
from src.core.logging import get_logger
from src.models.response import MetricsResponse
from src.storage.repository import BenchmarkRepository

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get metrics",
    description="Get aggregated statistics for all models and consensus.",
)
async def get_metrics(
    repository: BenchmarkRepository = Depends(get_benchmark_repository),
) -> MetricsResponse:
    logger.info("metrics_requested")

    # Get aggregated model stats
    model_stats = await repository.get_aggregated_model_stats()

    # Get aggregated consensus stats
    consensus_stats = await repository.get_aggregated_consensus_stats()

    # Format model stats for response
    formatted_models: dict[str, dict[str, float | int]] = {}
    for model, stats in model_stats.items():
        formatted_models[model] = {
            "total_runs": stats["total_runs"],
            "total_tasks": stats["total_tasks"],
            "accuracy": round(stats["accuracy"], 4),
            "avg_latency_ms": round(stats["avg_latency_ms"], 2),
            "avg_tokens": round(stats["avg_tokens"], 2),
        }

    return MetricsResponse(
        total_requests=consensus_stats["total_tasks"],
        models=formatted_models,
        consensus={
            "unanimous_rate": round(consensus_stats["unanimous_rate"], 4),
            "majority_rate": round(consensus_stats["majority_rate"], 4),
            "disagreement_rate": round(consensus_stats["disagreement_rate"], 4),
        },
    )
