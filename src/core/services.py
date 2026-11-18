"""Business logic services."""

from dataclasses import dataclass

from src.core.config import get_settings
from src.core.consensus import ConsensusAnalyzer, ConsensusAnalysisResult
from src.core.exceptions import ConsensusError
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
    4. Return unified result
    """

    def __init__(self, provider: ICompletionProvider) -> None:
        """Initialize service with LLM provider.

        Args:
            provider: LLM completion provider.
        """
        self._orchestrator = ModelOrchestrator(provider)
        self._parser = ResponseParser()
        self._analyzer = ConsensusAnalyzer()
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

        # Build response
        return self._build_response(
            processed_results=processed_results,
            consensus_result=consensus_result,
            total_latency_ms=orchestrator_result.total_latency_ms,
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
    ) -> ConsensusResponse:
        """Build final consensus response.

        Args:
            processed_results: Processed results from all models.
            consensus_result: Consensus analysis result.
            total_latency_ms: Total query latency.

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

        return ConsensusResponse(
            consensus=ConsensusResult(
                answer=consensus_result.answer,
                confidence=consensus_result.confidence,
                agreement=consensus_result.agreement,
                correct=None,  # Set by caller if known
            ),
            models=model_results,
            metadata={
                "total_latency_ms": round(total_latency_ms, 2),
                "consensus_method": consensus_result.method,
                "models_queried": len(processed_results),
                "models_successful": sum(
                    1 for r in processed_results.values() if r.solution is not None
                ),
            },
        )
