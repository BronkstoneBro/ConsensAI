"""Benchmark API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_benchmark_repository, get_benchmark_service
from src.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from src.core.exceptions import BenchmarkError
from src.core.logging import get_logger
from src.models.request import BenchmarkRequest
from src.models.response import BenchmarkModelStats, BenchmarkResponse, ConsensusStats
from src.storage.repository import BenchmarkRepository

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_200_OK,
    summary="Run benchmark",
    description="Run benchmark on a dataset of math tasks.",
)
async def run_benchmark(
    request: BenchmarkRequest,
    runner: BenchmarkRunner = Depends(get_benchmark_service),
    repository: BenchmarkRepository = Depends(get_benchmark_repository),
) -> BenchmarkResponse:
    logger.info(
        "benchmark_request_received",
        dataset=request.dataset,
        models=request.models,
        limit=request.limit,
    )

    try:
        config = BenchmarkConfig(
            dataset=request.dataset,
            models=request.models or [],
            limit=request.limit,
        )

        if not config.models:
            from src.core.config import get_settings
            config.models = get_settings().default_models

        run = await runner.run(config)

        await repository.save_benchmark_run(run)
        logger.info("benchmark_saved_to_db", benchmark_id=run.benchmark_id)

        # Convert to response
        results = {}
        for model, stats in run.stats.models.items():
            results[model] = BenchmarkModelStats(
                accuracy=stats.accuracy,
                avg_latency_ms=stats.avg_latency_ms,
                avg_tokens=stats.avg_tokens,
                total_tasks=stats.total_tasks,
                correct=stats.correct,
                incorrect=stats.incorrect,
                errors=stats.errors,
            )

        return BenchmarkResponse(
            benchmark_id=run.benchmark_id,
            status=run.status,
            dataset=run.dataset,
            results=results,
            consensus_stats=ConsensusStats(
                unanimous_agreement=run.stats.unanimous_agreement,
                majority_agreement=run.stats.majority_agreement,
                disagreement=run.stats.disagreement,
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
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        ) from e


@router.get(
    "/benchmark/list",
    summary="List benchmark runs",
    description="Get list of all benchmark runs.",
)
async def list_benchmarks(
    limit: int = 50,
    offset: int = 0,
    repository: BenchmarkRepository = Depends(get_benchmark_repository),
) -> list[dict]:
    runs = await repository.get_all_benchmark_runs(limit=limit, offset=offset)
    return [
        {
            "benchmark_id": run.benchmark_id,
            "dataset": run.dataset,
            "status": run.status,
            "total_tasks": run.total_tasks,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        for run in runs
    ]


@router.get(
    "/benchmark/{benchmark_id}",
    response_model=BenchmarkResponse,
    summary="Get benchmark status",
    description="Get the status and results of a benchmark run.",
)
async def get_benchmark(
    benchmark_id: str,
    repository: BenchmarkRepository = Depends(get_benchmark_repository),
) -> BenchmarkResponse:
    logger.info("benchmark_status_requested", benchmark_id=benchmark_id)

    run = await repository.get_benchmark_run(benchmark_id)

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Benchmark {benchmark_id} not found"},
        )

    model_stats_list = await repository.get_model_stats_for_run(run.id)

    results = {}
    for stats in model_stats_list:
        results[stats.model] = BenchmarkModelStats(
            accuracy=stats.accuracy,
            avg_latency_ms=stats.avg_latency_ms,
            avg_tokens=stats.avg_tokens,
            total_tasks=stats.total_tasks,
            correct=stats.correct,
            incorrect=stats.incorrect,
            errors=stats.errors,
        )

    return BenchmarkResponse(
        benchmark_id=run.benchmark_id,
        status=run.status,
        dataset=run.dataset,
        results=results,
        consensus_stats=ConsensusStats(
            unanimous_agreement=run.unanimous_agreement,
            majority_agreement=run.majority_agreement,
            disagreement=run.disagreement,
        ),
    )


@router.get(
    "/datasets",
    summary="List available datasets",
    description="Get list of available benchmark datasets.",
)
async def list_datasets() -> dict:
    from src.benchmark.datasets import AVAILABLE_DATASETS

    return {
        "datasets": AVAILABLE_DATASETS,
        "description": {
            "arithmetic": "Basic arithmetic operations (easy)",
            "algebra": "Algebraic equations (medium)",
            "logic": "Word problems and logical reasoning (hard)",
        },
    }
