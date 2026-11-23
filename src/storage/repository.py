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
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_benchmark_run(self, run: BenchmarkRun) -> None:
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
        result = await self._session.execute(
            select(BenchmarkRunModel)
            .order_by(BenchmarkRunModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_model_stats_for_run(self, benchmark_run_id: int) -> list[ModelStatsModel]:
        result = await self._session.execute(
            select(ModelStatsModel).where(
                ModelStatsModel.benchmark_run_id == benchmark_run_id
            )
        )
        return list(result.scalars().all())

    async def get_aggregated_model_stats(self) -> dict[str, dict]:
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

    async def get_aggregated_consensus_stats(self) -> dict[str, float | int]:
        result = await self._session.execute(select(BenchmarkRunModel))
        all_runs = result.scalars().all()

        total_unanimous = 0
        total_majority = 0
        total_disagreement = 0
        total_tasks = 0

        for run in all_runs:
            total_unanimous += run.unanimous_agreement
            total_majority += run.majority_agreement
            total_disagreement += run.disagreement
            total_tasks += run.total_tasks

        # Calculate rates
        if total_tasks > 0:
            unanimous_rate = total_unanimous / total_tasks
            majority_rate = total_majority / total_tasks
            disagreement_rate = total_disagreement / total_tasks
        else:
            unanimous_rate = 0.0
            majority_rate = 0.0
            disagreement_rate = 0.0

        return {
            "total_tasks": total_tasks,
            "total_benchmark_runs": len(all_runs),
            "unanimous_agreement": total_unanimous,
            "majority_agreement": total_majority,
            "disagreement": total_disagreement,
            "unanimous_rate": unanimous_rate,
            "majority_rate": majority_rate,
            "disagreement_rate": disagreement_rate,
        }
