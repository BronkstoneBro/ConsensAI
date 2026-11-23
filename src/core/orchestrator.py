"""Model orchestrator for parallel LLM queries."""

import asyncio
from dataclasses import dataclass, field

from src.core.config import get_settings
from src.core.exceptions import LLMProviderError
from src.core.logging import get_logger
from src.llm.providers import ICompletionProvider, LLMResponse

logger = get_logger(__name__)


@dataclass
class ModelQueryResult:
    model: str
    success: bool
    response: LLMResponse | None = None
    error: str | None = None


@dataclass
class OrchestratorResult:
    results: dict[str, ModelQueryResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    successful_count: int = 0
    failed_count: int = 0

    @property
    def successful_responses(self) -> dict[str, LLMResponse]:
        return {
            model: result.response
            for model, result in self.results.items()
            if result.success and result.response is not None
        }


class ModelOrchestrator:
    """Sends the same prompt to multiple models concurrently."""

    def __init__(self, provider: ICompletionProvider) -> None:
        self._provider = provider
        self._settings = get_settings()

    async def query_models(
        self,
        prompt: str,
        models: list[str] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> OrchestratorResult:
        if models is None:
            models = self._settings.default_models

        logger.info(
            "orchestrator_query_started",
            models=models,
            prompt_length=len(prompt),
        )

        # Create tasks for parallel execution
        tasks = [
            self._query_single_model(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for model in models
        ]

        # Execute all queries in parallel
        import time
        start_time = time.perf_counter()
        query_results = await asyncio.gather(*tasks, return_exceptions=False)
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Build result
        result = OrchestratorResult(total_latency_ms=total_latency_ms)

        for model, query_result in zip(models, query_results):
            result.results[model] = query_result
            if query_result.success:
                result.successful_count += 1
            else:
                result.failed_count += 1

        logger.info(
            "orchestrator_query_completed",
            total_latency_ms=round(total_latency_ms, 2),
            successful=result.successful_count,
            failed=result.failed_count,
        )

        return result

    async def _query_single_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ModelQueryResult:
        try:
            response = await self._provider.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return ModelQueryResult(
                model=model,
                success=True,
                response=response,
            )

        except LLMProviderError as e:
            logger.warning(
                "model_query_failed",
                model=model,
                error=str(e),
            )
            return ModelQueryResult(
                model=model,
                success=False,
                error=str(e),
            )

        except Exception as e:
            logger.error(
                "model_query_unexpected_error",
                model=model,
                error=str(e),
            )
            return ModelQueryResult(
                model=model,
                success=False,
                error=f"Unexpected error: {e}",
            )
