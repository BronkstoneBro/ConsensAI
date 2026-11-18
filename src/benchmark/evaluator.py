"""Evaluation logic for benchmark results."""

from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.models.math_task import MathTask

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """Result of a single task evaluation."""

    task_id: str
    question: str
    correct_answer: float
    model_answer: float | None
    is_correct: bool
    latency_ms: float
    tokens_used: int
    error: str | None = None


@dataclass
class ModelStats:
    """Aggregated statistics for a single model."""

    model: str
    total_tasks: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as fraction of correct answers out of total tasks."""
        return self.correct / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency per task."""
        return self.total_latency_ms / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens per task."""
        return self.total_tokens / self.total_tasks if self.total_tasks > 0 else 0.0


@dataclass
class BenchmarkStats:
    """Overall benchmark statistics."""

    total_tasks: int = 0
    models: dict[str, ModelStats] = field(default_factory=dict)
    unanimous_agreement: int = 0
    majority_agreement: int = 0
    disagreement: int = 0

    @property
    def consensus_accuracy(self) -> float:
        """Calculate consensus accuracy (unanimous + majority correct)."""
        total = self.unanimous_agreement + self.majority_agreement + self.disagreement
        return (self.unanimous_agreement + self.majority_agreement) / total if total > 0 else 0.0


class Evaluator:
    """Evaluates model answers against correct answers."""

    def __init__(self, tolerance: float = 0.01) -> None:
        """Initialize evaluator.

        Args:
            tolerance: Relative tolerance for comparing answers.
        """
        self._tolerance = tolerance

    def is_correct(self, model_answer: float | None, correct_answer: float) -> bool:
        """Check if model answer matches correct answer.

        Args:
            model_answer: Answer from the model.
            correct_answer: Expected correct answer.

        Returns:
            True if answers match within tolerance.
        """
        if model_answer is None:
            return False

        # Handle zero case
        if correct_answer == 0:
            return abs(model_answer) < self._tolerance

        # Relative comparison
        relative_diff = abs(model_answer - correct_answer) / abs(correct_answer)
        return relative_diff <= self._tolerance

    def evaluate_task(
        self,
        task: MathTask,
        model_answer: float | None,
        latency_ms: float,
        tokens_used: int,
        error: str | None = None,
    ) -> TaskResult:
        """Evaluate a single task result.

        Args:
            task: The math task.
            model_answer: Model's answer.
            latency_ms: Response latency.
            tokens_used: Tokens consumed.
            error: Error message if any.

        Returns:
            TaskResult with evaluation.
        """
        is_correct = self.is_correct(model_answer, task.correct_answer)

        result = TaskResult(
            task_id=task.id,
            question=task.question,
            correct_answer=task.correct_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            error=error,
        )

        logger.debug(
            "task_evaluated",
            task_id=task.id,
            correct=is_correct,
            model_answer=model_answer,
            expected=task.correct_answer,
        )

        return result

    def aggregate_results(
        self,
        model: str,
        results: list[TaskResult],
    ) -> ModelStats:
        """Aggregate results into model statistics.

        Args:
            model: Model identifier.
            results: List of task results.

        Returns:
            ModelStats with aggregated metrics.
        """
        stats = ModelStats(model=model, total_tasks=len(results))

        for result in results:
            if result.error:
                stats.errors += 1
            elif result.is_correct:
                stats.correct += 1
            else:
                stats.incorrect += 1

            stats.total_latency_ms += result.latency_ms
            stats.total_tokens += result.tokens_used

        logger.info(
            "model_stats_aggregated",
            model=model,
            accuracy=round(stats.accuracy, 3),
            total=stats.total_tasks,
            correct=stats.correct,
        )

        return stats
