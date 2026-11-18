"""Repository for database operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.benchmark.evaluator import BenchmarkStats, ModelStats, TaskResult
from src.benchmark.runner import BenchmarkRun
from src.core.logging import get_logger
from src.storage.database import (
    BenchmarkRunModel,
    ModelStatsModel,
    TaskResultModel,
)

logger = get_logger(__name__)


class BenchmarkRepository:
    """Repository for benchmark data operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def save_benchmark_run(self, run: BenchmarkRun) -> None:
        """Save benchmark run to database.

        Args:
            run: Benchmark run to save.
        """
        # Create benchmark run record
        db_run = BenchmarkRunModel(
            benchmark_id=run.benchmark_id,
            dataset=run.dataset,
            models=run.models,
            status=run.status,
            total_tasks=run.stats.total_tasks,
            unanimous_agreement=run.stats.unanimous_agreement,
            majority_agreement=run.stats.majority_agreement,
            disagreement=run.stats.disagreement,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

        self._session.add(db_run)
        await self._session.flush()

        # Save model stats
        for model, stats in run.stats.models.items():
            db_stats = ModelStatsModel(
                benchmark_run_id=db_run.id,
                model=model,
                total_tasks=stats.total_tasks,
                correct=stats.correct,
                incorrect=stats.incorrect,
                errors=stats.errors,
                total_latency_ms=stats.total_latency_ms,
                total_tokens=stats.total_tokens,
                accuracy=stats.accuracy,
                avg_latency_ms=stats.avg_latency_ms,
                avg_tokens=stats.avg_tokens,
            )
            self._session.add(db_stats)

        # Save task results
        for model, results in run.task_results.items():
            for result in results:
                db_result = TaskResultModel(
                    benchmark_run_id=db_run.id,
                    model=model,
                    task_id=result.task_id,
                    question=result.question,
                    correct_answer=result.correct_answer,
                    model_answer=result.model_answer,
                    is_correct=result.is_correct,
                    latency_ms=result.latency_ms,
                    tokens_used=result.tokens_used,
                    error=result.error,
                )
                self._session.add(db_result)

        await self._session.commit()

        logger.info(
            "benchmark_run_saved",
            benchmark_id=run.benchmark_id,
            status=run.status,
        )

    async def get_benchmark_run(self, benchmark_id: str) -> BenchmarkRunModel | None:
        """Get benchmark run by ID.

        Args:
            benchmark_id: Benchmark identifier.

        Returns:
            Benchmark run record or None.
        """
        result = await self._session.execute(
            select(BenchmarkRunModel).where(
                BenchmarkRunModel.benchmark_id == benchmark_id
            )
        )
        return result.scalar_one_or_none()

    async def get_all_benchmark_runs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BenchmarkRunModel]:
        """Get all benchmark runs.

        Args:
            limit: Maximum number of records.
            offset: Number of records to skip.

        Returns:
            List of benchmark runs.
        """
        result = await self._session.execute(
            select(BenchmarkRunModel)
            .order_by(BenchmarkRunModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_model_stats_for_run(
        self,
        benchmark_run_id: int,
    ) -> list[ModelStatsModel]:
        """Get model stats for a benchmark run.

        Args:
            benchmark_run_id: Benchmark run database ID.

        Returns:
            List of model stats.
        """
        result = await self._session.execute(
            select(ModelStatsModel).where(
                ModelStatsModel.benchmark_run_id == benchmark_run_id
            )
        )
        return list(result.scalars().all())

    async def get_aggregated_model_stats(self) -> dict[str, dict]:
        """Get aggregated statistics across all benchmark runs.

        Returns:
            Dict mapping model name to aggregated stats.
        """
        result = await self._session.execute(select(ModelStatsModel))
        all_stats = result.scalars().all()

        aggregated: dict[str, dict] = {}

        for stats in all_stats:
            if stats.model not in aggregated:
                aggregated[stats.model] = {
                    "total_runs": 0,
                    "total_tasks": 0,
                    "total_correct": 0,
                    "total_latency_ms": 0.0,
                    "total_tokens": 0,
                }

            agg = aggregated[stats.model]
            agg["total_runs"] += 1
            agg["total_tasks"] += stats.total_tasks
            agg["total_correct"] += stats.correct
            agg["total_latency_ms"] += stats.total_latency_ms
            agg["total_tokens"] += stats.total_tokens

        # Calculate averages
        for model, agg in aggregated.items():
            if agg["total_tasks"] > 0:
                agg["accuracy"] = agg["total_correct"] / agg["total_tasks"]
                agg["avg_latency_ms"] = agg["total_latency_ms"] / agg["total_tasks"]
                agg["avg_tokens"] = agg["total_tokens"] / agg["total_tasks"]
            else:
                agg["accuracy"] = 0.0
                agg["avg_latency_ms"] = 0.0
                agg["avg_tokens"] = 0.0

        return aggregated
