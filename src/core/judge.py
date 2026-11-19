"""Judge model service for resolving disagreements."""

import json
import re
from dataclasses import dataclass

from src.core.config import get_settings
from src.core.logging import get_logger
from src.llm.prompts import JUDGE_SYSTEM_PROMPT, format_judge_prompt
from src.llm.providers import ICompletionProvider
from src.models.math_task import MathSolution

logger = get_logger(__name__)


@dataclass
class JudgeDecision:
    """Result of judge model evaluation."""

    correct_answer: float | None
    winning_model: str | None
    analysis: list[str]
    reasoning: str
    judge_model: str
    latency_ms: float
    success: bool
    error: str | None = None


class JudgeService:
    """Service for using a judge model to resolve disagreements.

    When models disagree on an answer, the judge analyzes all responses
    and determines which answer is most likely correct.
    """

    def __init__(
        self,
        provider: ICompletionProvider,
        judge_model: str | None = None,
    ) -> None:
        """Initialize judge service.

        Args:
            provider: LLM completion provider.
            judge_model: Model to use as judge. Defaults to settings.judge_model.
        """
        self._provider = provider
        self._settings = get_settings()
        self._judge_model = judge_model or self._settings.judge_model

    async def judge(
        self,
        question: str,
        solutions: dict[str, MathSolution],
    ) -> JudgeDecision:
        """Ask judge model to evaluate multiple solutions.

        Args:
            question: Original math question.
            solutions: Dict mapping model name to its MathSolution.

        Returns:
            JudgeDecision with the judge's verdict.
        """
        logger.info(
            "judge_evaluation_started",
            question_length=len(question),
            num_solutions=len(solutions),
            judge_model=self._judge_model,
        )

        # Format model answers for the prompt
        model_answers = self._format_solutions(solutions)
        prompt = format_judge_prompt(question, model_answers)

        try:
            # Query judge model
            response = await self._provider.complete(
                prompt=prompt,
                model=self._judge_model,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )

            # Parse judge response
            decision = self._parse_judge_response(
                response.content,
                self._judge_model,
                response.latency_ms,
            )

            logger.info(
                "judge_evaluation_completed",
                correct_answer=decision.correct_answer,
                winning_model=decision.winning_model,
                latency_ms=decision.latency_ms,
            )

            return decision

        except Exception as e:
            logger.error("judge_evaluation_failed", error=str(e))
            return JudgeDecision(
                correct_answer=None,
                winning_model=None,
                analysis=[],
                reasoning=f"Judge evaluation failed: {e}",
                judge_model=self._judge_model,
                latency_ms=0.0,
                success=False,
                error=str(e),
            )

    def _format_solutions(
        self,
        solutions: dict[str, MathSolution],
    ) -> dict[str, str]:
        """Format solutions for judge prompt.

        Args:
            solutions: Dict mapping model name to MathSolution.

        Returns:
            Dict mapping model name to formatted answer string.
        """
        formatted = {}
        for model, solution in solutions.items():
            steps_str = " -> ".join(solution.steps) if solution.steps else "no steps"
            formatted[model] = (
                f"Answer: {solution.answer}, "
                f"Method: {solution.method}, "
                f"Steps: {steps_str}"
            )
        return formatted

    def _parse_judge_response(
        self,
        content: str,
        judge_model: str,
        latency_ms: float,
    ) -> JudgeDecision:
        """Parse judge model response.

        Args:
            content: Raw response from judge model.
            judge_model: Model used as judge.
            latency_ms: Response latency.

        Returns:
            JudgeDecision parsed from response.
        """
        try:
            # Try to parse JSON directly
            data = self._extract_json(content)

            return JudgeDecision(
                correct_answer=data.get("correct_answer"),
                winning_model=data.get("winning_model"),
                analysis=data.get("analysis", []),
                reasoning=data.get("reasoning", ""),
                judge_model=judge_model,
                latency_ms=latency_ms,
                success=True,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("judge_response_parse_failed", error=str(e))

            # Fallback: try to extract answer from text
            answer = self._extract_number_from_text(content)

            return JudgeDecision(
                correct_answer=answer,
                winning_model=None,
                analysis=[],
                reasoning=content[:500],
                judge_model=judge_model,
                latency_ms=latency_ms,
                success=answer is not None,
                error="Failed to parse structured response" if answer is None else None,
            )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from response content.

        Args:
            content: Raw response text.

        Returns:
            Parsed JSON dict.

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed.
        """
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise json.JSONDecodeError("No JSON found", content, 0)

    def _extract_number_from_text(self, content: str) -> float | None:
        """Extract a number from text as fallback.

        Args:
            content: Text to extract number from.

        Returns:
            Extracted number or None.
        """
        # Look for patterns like "correct answer is X" or "answer: X"
        patterns = [
            r"correct[_ ]answer[:\s]+(-?\d+\.?\d*)",
            r"answer[:\s]+(-?\d+\.?\d*)",
            r"result[:\s]+(-?\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None
