"""Benchmark runner for evaluating models on math tasks."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from src.benchmark.datasets import load_dataset
from src.benchmark.evaluator import BenchmarkStats, Evaluator, ModelStats, TaskResult
from src.core.config import get_settings
from src.core.consensus import ConsensusAnalyzer
from src.core.logging import get_logger
from src.core.orchestrator import ModelOrchestrator
from src.llm.prompts import MATH_SYSTEM_PROMPT, format_math_prompt
from src.llm.providers import ICompletionProvider
from src.llm.structured import ResponseParser
from src.models.math_task import MathTask
from src.models.response import AgreementLevel

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    dataset: str
    models: list[str]
    limit: int | None = None
    batch_size: int = 5


@dataclass
class BenchmarkRun:
    """Results of a benchmark run."""

    benchmark_id: str
    dataset: str
    models: list[str]
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    stats: BenchmarkStats = field(default_factory=BenchmarkStats)
    task_results: dict[str, list[TaskResult]] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs benchmarks on math task datasets."""

    def __init__(self, provider: ICompletionProvider) -> None:
        """Initialize runner with LLM provider.

        Args:
            provider: LLM completion provider.
        """
        self._orchestrator = ModelOrchestrator(provider)
        self._parser = ResponseParser()
        self._evaluator = Evaluator()
        self._analyzer = ConsensusAnalyzer()
        self._settings = get_settings()

    async def run(self, config: BenchmarkConfig) -> BenchmarkRun:
        """Run benchmark with given configuration.

        Args:
            config: Benchmark configuration.

        Returns:
            BenchmarkRun with results.
        """
        benchmark_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info(
            "benchmark_started",
            benchmark_id=benchmark_id,
            dataset=config.dataset,
            models=config.models,
        )

        run = BenchmarkRun(
            benchmark_id=benchmark_id,
            dataset=config.dataset,
            models=config.models,
            started_at=datetime.now(),
        )

        # Initialize task results for each model
        for model in config.models:
            run.task_results[model] = []

        try:
            # Load tasks
            tasks = load_dataset(config.dataset, limit=config.limit)
            run.stats.total_tasks = len(tasks)

            # Process in batches
            batch_size = config.batch_size
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                await self._process_batch(batch, config.models, run)

            # Aggregate statistics
            for model in config.models:
                model_stats = self._evaluator.aggregate_results(
                    model, run.task_results[model]
                )
                run.stats.models[model] = model_stats

            run.status = "completed"
            run.completed_at = datetime.now()

            logger.info(
                "benchmark_completed",
                benchmark_id=benchmark_id,
                total_tasks=run.stats.total_tasks,
                unanimous=run.stats.unanimous_agreement,
            )

        except Exception as e:
            run.status = "failed"
            run.completed_at = datetime.now()
            logger.error("benchmark_failed", benchmark_id=benchmark_id, error=str(e))
            raise

        return run

    async def _process_batch(
        self,
        tasks: list[MathTask],
        models: list[str],
        run: BenchmarkRun,
    ) -> None:
        """Process a batch of tasks.

        Args:
            tasks: Batch of tasks to process.
            models: Models to evaluate.
            run: Benchmark run to update.
        """
        for task in tasks:
            await self._process_task(task, models, run)

    async def _process_task(
        self,
        task: MathTask,
        models: list[str],
        run: BenchmarkRun,
    ) -> None:
        """Process a single task across all models.

        Args:
            task: Task to process.
            models: Models to evaluate.
            run: Benchmark run to update.
        """
        prompt = format_math_prompt(task.question)

        # Query all models
        result = await self._orchestrator.query_models(
            prompt=prompt,
            models=models,
            system_prompt=MATH_SYSTEM_PROMPT,
            temperature=0.0,
        )

        # Collect answers for consensus
        model_answers: dict[str, float | None] = {}

        # Process each model's response
        for model, query_result in result.results.items():
            if query_result.success and query_result.response:
                response = query_result.response
                solution = self._parser.parse_math_solution(response.content)

                answer = solution.answer if solution else None
                model_answers[model] = answer

                task_result = self._evaluator.evaluate_task(
                    task=task,
                    model_answer=answer,
                    latency_ms=response.latency_ms,
                    tokens_used=response.tokens_prompt + response.tokens_completion,
                )
            else:
                model_answers[model] = None
                task_result = self._evaluator.evaluate_task(
                    task=task,
                    model_answer=None,
                    latency_ms=0,
                    tokens_used=0,
                    error=query_result.error,
                )

            run.task_results[model].append(task_result)

        # Analyze consensus
        self._update_consensus_stats(model_answers, task.correct_answer, run)

    def _update_consensus_stats(
        self,
        model_answers: dict[str, float | None],
        correct_answer: float,
        run: BenchmarkRun,
    ) -> None:
        """Update consensus statistics.

        Args:
            model_answers: Answers from each model.
            correct_answer: Expected correct answer.
            run: Benchmark run to update.
        """
        # Filter out None answers
        valid_answers = {m: a for m, a in model_answers.items() if a is not None}

        if not valid_answers:
            run.stats.disagreement += 1
            return

        # Check agreement
        answer_values = list(valid_answers.values())
        unique_answers = set(round(a, 2) for a in answer_values)

        if len(unique_answers) == 1:
            # All models agree
            run.stats.unanimous_agreement += 1
        elif len(unique_answers) < len(answer_values):
            # Some models agree (majority)
            run.stats.majority_agreement += 1
        else:
            # All different
            run.stats.disagreement += 1
