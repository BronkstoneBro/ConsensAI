"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "ConsensAI"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM Providers
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: str = Field(default="", description="Groq API key (free tier)")

    # Default models for consensus
    # Available Groq models: openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b
    default_models: list[str] = Field(
        default=["groq/openai/gpt-oss-20b", "groq/qwen/qwen3.8-27b"],
        description="List of models to use for consensus",
    )

    # Consensus settings
    consensus_timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Judge model for resolving disagreements
    judge_model: str = Field(
        default="groq/llama-3.3-70b-versatile",
        description="Model to use as judge when models disagree",
    )
    use_judge_on_disagreement: bool = Field(
        default=True,
        description="Whether to use judge model when models disagree",
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/consensai.db"

    # Benchmark
    benchmark_batch_size: int = 10


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
