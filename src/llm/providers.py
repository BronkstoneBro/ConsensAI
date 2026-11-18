"""LLM providers abstraction using LiteLLM."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import acompletion

from src.core.config import get_settings
from src.core.exceptions import LLMProviderError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM provider."""

    model: str
    content: str
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    raw_response: dict[str, Any] | None = None


class ICompletionProvider(ABC):
    """Interface for LLM completion providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate completion for the given prompt.

        Args:
            prompt: User prompt to complete.
            model: Model identifier (e.g., 'ollama/llama3.2').
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            LLMProviderError: If completion fails.
        """


class LiteLLMProvider(ICompletionProvider):
    """LiteLLM-based provider supporting multiple backends."""

    def __init__(self) -> None:
        """Initialize LiteLLM provider."""
        settings = get_settings()

        # Configure LiteLLM
        litellm.drop_params = True
        litellm.set_verbose = settings.debug

        # Set API keys from settings
        if settings.groq_api_key:
            litellm.api_key = settings.groq_api_key

        self._ollama_base_url = settings.ollama_base_url
        self._timeout = settings.consensus_timeout_seconds

    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate completion using LiteLLM.

        Args:
            prompt: User prompt to complete.
            model: Model identifier (e.g., 'ollama/llama3.2').
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            LLMProviderError: If completion fails.
        """
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self._timeout,
        }

        # Add Ollama base URL if it's an Ollama model
        if model.startswith("ollama/"):
            kwargs["api_base"] = self._ollama_base_url

        logger.debug(
            "llm_request_started",
            model=model,
            prompt_length=len(prompt),
        )

        start_time = time.perf_counter()

        try:
            response = await acompletion(**kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content or ""
            usage = response.usage

            logger.debug(
                "llm_request_completed",
                model=model,
                latency_ms=round(latency_ms, 2),
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
            )

            return LLMResponse(
                model=model,
                content=content,
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except litellm.exceptions.Timeout as e:
            logger.warning("llm_request_timeout", model=model, error=str(e))
            raise LLMProviderError(model, f"Request timed out: {e}") from e

        except litellm.exceptions.APIConnectionError as e:
            logger.warning("llm_connection_error", model=model, error=str(e))
            raise LLMProviderError(model, f"Connection failed: {e}") from e

        except Exception as e:
            logger.error("llm_request_failed", model=model, error=str(e))
            raise LLMProviderError(model, f"Unexpected error: {e}") from e
