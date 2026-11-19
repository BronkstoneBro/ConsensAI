"""Business logic services."""

from dataclasses import dataclass, field

from src.core.config import get_settings
from src.core.consensus import ConsensusAnalyzer, ConsensusAnalysisResult
from src.core.exceptions import ConsensusError
from src.core.judge import JudgeDecision, JudgeService
from src.core.logging import get_logger
from src.core.orchestrator import ModelOrchestrator, OrchestratorResult
from src.llm.prompts import MATH_SYSTEM_PROMPT, format_math_prompt
from src.llm.providers import ICompletionProvider
from src.llm.structured import ResponseParser
from src.models.math_task import MathSolution
from src.models.request import ConsensusRequest
from src.models.response import (
    AgreementLevel,
    ConsensusResponse,
    ConsensusResult,
    ModelResult,
)

logger = get_logger(__name__)


@dataclass
class ProcessedModelResult:
    """Processed result from a single model."""

    model: str
    solution: MathSolution | None
    raw_response: str
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    error: str | None = None


class ConsensusService:
    """Service for consensus-based question answering.

    Orchestrates the full flow:
    1. Send question to multiple models
    2. Parse structured responses
    3. Analyze consensus
    4. Use judge model if disagreement (optional)
    5. Return unified result
    """

    def __init__(self, provider: ICompletionProvider) -> None:
        """Initialize service with LLM provider.

        Args:
            provider: LLM completion provider.
        """
        self._provider = provider
        self._orchestrator = ModelOrchestrator(provider)
        self._parser = ResponseParser()
        self._analyzer = ConsensusAnalyzer()
        self._judge = JudgeService(provider)
        self._settings = get_settings()

    async def analyze(self, request: ConsensusRequest) -> ConsensusResponse:
        """Analyze question using multiple models and determine consensus.

        Args:
            request: Consensus request with question and optional model list.

        Returns:
            ConsensusResponse with consensus result and per-model details.

        Raises:
            ConsensusError: If no valid responses received.
        """
        logger.info(
            "consensus_service_started",
            question_length=len(request.question),
            models=request.models,
            use_judge=request.use_judge,
        )

        # Format prompt
        prompt = format_math_prompt(request.question)
        system_prompt = MATH_SYSTEM_PROMPT if request.require_structured else None

        # Query all models
        orchestrator_result = await self._orchestrator.query_models(
            prompt=prompt,
            models=request.models,
            system_prompt=system_prompt,
            temperature=request.temperature,
        )

        # Process responses
        processed_results = self._process_responses(orchestrator_result)

        # Extract valid solutions for consensus
        valid_solutions = {
            model: result.solution
            for model, result in processed_results.items()
            if result.solution is not None
        }

        if not valid_solutions:
            logger.warning("no_valid_solutions")
            raise ConsensusError("No valid responses from any model")

        # Analyze consensus
        consensus_result = self._analyzer.auto_analyze(valid_solutions)

        # Use judge if disagreement and judge is enabled
        judge_decision: JudgeDecision | None = None
        if self._should_use_judge(consensus_result, request.use_judge):
            judge_decision = await self._judge.judge(
                question=request.question,
                solutions=valid_solutions,
            )
            # Update consensus result based on judge decision
            if judge_decision.success and judge_decision.correct_answer is not None:
                consensus_result = self._apply_judge_decision(
                    consensus_result, judge_decision
                )

        # Build response
        return self._build_response(
            processed_results=processed_results,
            consensus_result=consensus_result,
            total_latency_ms=orchestrator_result.total_latency_ms,
            judge_decision=judge_decision,
        )

    def _should_use_judge(
        self,
        consensus_result: ConsensusAnalysisResult,
        use_judge: bool,
    ) -> bool:
        """Determine if judge should be used.

        Args:
            consensus_result: Initial consensus analysis result.
            use_judge: User preference for using judge.

        Returns:
            True if judge should be called.
        """
        if not use_judge:
            return False

        if not self._settings.use_judge_on_disagreement:
            return False

        # Use judge for disagreement or split
        return consensus_result.agreement in (
            AgreementLevel.DISAGREEMENT,
            AgreementLevel.SPLIT,
        )

    def _apply_judge_decision(
        self,
        consensus_result: ConsensusAnalysisResult,
        judge_decision: JudgeDecision,
    ) -> ConsensusAnalysisResult:
        """Update consensus result based on judge decision.

        Args:
            consensus_result: Original consensus result.
            judge_decision: Judge's decision.

        Returns:
            Updated ConsensusAnalysisResult.
        """
        logger.info(
            "applying_judge_decision",
            original_answer=consensus_result.answer,
            judge_answer=judge_decision.correct_answer,
        )

        return ConsensusAnalysisResult(
            answer=judge_decision.correct_answer,
            confidence=0.8,  # High confidence since judge resolved it
            agreement=AgreementLevel.MAJORITY,  # Treat judge decision as majority
            method="judge_model",
            vote_distribution=consensus_result.vote_distribution,
        )

    def _process_responses(
        self,
        orchestrator_result: OrchestratorResult,
    ) -> dict[str, ProcessedModelResult]:
        """Process raw responses into structured results.

        Args:
            orchestrator_result: Raw results from orchestrator.

        Returns:
            Dict mapping model name to processed result.
        """
        processed: dict[str, ProcessedModelResult] = {}

        for model, query_result in orchestrator_result.results.items():
            if not query_result.success or query_result.response is None:
                processed[model] = ProcessedModelResult(
                    model=model,
                    solution=None,
                    raw_response="",
                    latency_ms=0.0,
                    tokens_prompt=0,
                    tokens_completion=0,
                    error=query_result.error,
                )
                continue

            response = query_result.response

            # Parse structured solution
            solution = self._parser.parse_math_solution(response.content)

            processed[model] = ProcessedModelResult(
                model=model,
                solution=solution,
                raw_response=response.content,
                latency_ms=response.latency_ms,
                tokens_prompt=response.tokens_prompt,
                tokens_completion=response.tokens_completion,
                error=None if solution else "Failed to parse response",
            )

        return processed

    def _build_response(
        self,
        processed_results: dict[str, ProcessedModelResult],
        consensus_result: ConsensusAnalysisResult,
        total_latency_ms: float,
        judge_decision: JudgeDecision | None = None,
    ) -> ConsensusResponse:
        """Build final consensus response.

        Args:
            processed_results: Processed results from all models.
            consensus_result: Consensus analysis result.
            total_latency_ms: Total query latency.
            judge_decision: Optional judge decision if used.

        Returns:
            ConsensusResponse for API.
        """
        # Build model results dict
        model_results: dict[str, ModelResult] = {}

        for model, result in processed_results.items():
            model_results[model] = ModelResult(
                model=model,
                answer=result.solution.answer if result.solution else None,
                raw_response=result.raw_response,
                steps=result.solution.steps if result.solution else None,
                latency_ms=result.latency_ms,
                tokens_prompt=result.tokens_prompt,
                tokens_completion=result.tokens_completion,
                error=result.error,
            )

        # Build metadata
        metadata: dict[str, float | int | str | bool] = {
            "total_latency_ms": round(total_latency_ms, 2),
            "consensus_method": consensus_result.method,
            "models_queried": len(processed_results),
            "models_successful": sum(
                1 for r in processed_results.values() if r.solution is not None
            ),
        }

        # Add judge information if used
        if judge_decision is not None:
            metadata["judge_used"] = True
            metadata["judge_model"] = judge_decision.judge_model
            metadata["judge_latency_ms"] = round(judge_decision.latency_ms, 2)
            metadata["judge_success"] = judge_decision.success
            if judge_decision.reasoning:
                metadata["judge_reasoning"] = judge_decision.reasoning[:200]
        else:
            metadata["judge_used"] = False

        return ConsensusResponse(
            consensus=ConsensusResult(
                answer=consensus_result.answer,
                confidence=consensus_result.confidence,
                agreement=consensus_result.agreement,
                correct=None,  # Set by caller if known
            ),
            models=model_results,
            metadata=metadata,
        )
