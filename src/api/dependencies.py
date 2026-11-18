"""FastAPI dependency injection configuration."""

from functools import lru_cache

from src.benchmark.runner import BenchmarkRunner
from src.core.config import Settings, get_settings
from src.core.services import ConsensusService
from src.llm.providers import ICompletionProvider, LiteLLMProvider


@lru_cache
def get_llm_provider() -> ICompletionProvider:
    """Get LLM provider instance (cached)."""
    return LiteLLMProvider()


@lru_cache
def get_consensus_service() -> ConsensusService:
    """Get consensus service instance (cached)."""
    provider = get_llm_provider()
    return ConsensusService(provider)


@lru_cache
def get_benchmark_service() -> BenchmarkRunner:
    """Get benchmark runner instance (cached)."""
    provider = get_llm_provider()
    return BenchmarkRunner(provider)
