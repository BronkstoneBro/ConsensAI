"""FastAPI dependency injection configuration."""

from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.benchmark.runner import BenchmarkRunner
from src.core.config import Settings, get_settings
from src.core.services import ConsensusService
from src.llm.providers import ICompletionProvider, LiteLLMProvider
from src.storage.database import get_session_factory
from src.storage.repository import BenchmarkRepository


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


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Get database session for request scope."""
    factory = await get_session_factory()
    async with factory() as session:
        yield session


async def get_benchmark_repository(
    session: AsyncSession = Depends(get_db_session),
) -> BenchmarkRepository:
    """Get benchmark repository instance."""
    return BenchmarkRepository(session)
