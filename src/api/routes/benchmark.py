"""Benchmark API endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.core.exceptions import BenchmarkError
from src.core.logging import get_logger
from src.models.request import BenchmarkRequest
from src.models.response import BenchmarkModelStats, BenchmarkResponse, ConsensusStats

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_200_OK,
    summary="Run benchmark",
    description="Run benchmark on a dataset of math tasks.",
)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run benchmark on specified dataset.

    Evaluates multiple models on math tasks and returns accuracy,
    latency, and consensus statistics.
    """
    logger.info(
        "benchmark_request_received",
        dataset=request.dataset,
        models=request.models,
        limit=request.limit,
    )

    try:
        # TODO: Implement actual benchmark in Phase 3
        # For now, return a placeholder response

        return BenchmarkResponse(
            benchmark_id="placeholder_benchmark",
            status="not_implemented",
            dataset=request.dataset,
            results={},
            consensus_stats=ConsensusStats(
                unanimous_agreement=0,
                majority_agreement=0,
                disagreement=0,
            ),
        )

    except BenchmarkError as e:
        logger.warning("benchmark_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "BENCHMARK_ERROR", "message": str(e)},
        ) from e

    except Exception as e:
        logger.error("benchmark_unexpected_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
        ) from e


@router.get(
    "/benchmark/{benchmark_id}",
    response_model=BenchmarkResponse,
    summary="Get benchmark status",
    description="Get the status and results of a benchmark run.",
)
async def get_benchmark(benchmark_id: str) -> BenchmarkResponse:
    """Get benchmark status and results by ID."""
    logger.info("benchmark_status_requested", benchmark_id=benchmark_id)

    # TODO: Implement in Phase 3
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "NOT_FOUND", "message": f"Benchmark {benchmark_id} not found"},
    )
