"""Custom exceptions for ConsensAI."""


class ConsensAIError(Exception):
    """Base class for all ConsensAI errors."""


class ValidationError(ConsensAIError):
    """Input validation error."""


class LLMProviderError(ConsensAIError):
    """Error when communicating with LLM provider."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ConsensusError(ConsensAIError):
    """Failed to reach consensus among models."""


class RepositoryError(ConsensAIError):
    """Database operation error."""


class BenchmarkError(ConsensAIError):
    """Benchmark execution error."""


class ConfigurationError(ConsensAIError):
    """Application configuration error."""
