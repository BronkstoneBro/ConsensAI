"""Request models for API endpoints."""

from pydantic import BaseModel, Field


class ConsensusRequest(BaseModel):
    """Request for consensus analysis."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question or task to send to models",
    )
    models: list[str] | None = Field(
        default=None,
        description="List of models to query. If None, uses default models from config.",
    )
    require_structured: bool = Field(
        default=True,
        description="Whether to require structured JSON output from models",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Сколько будет 15% от 80?",
                "models": ["ollama/llama3.2", "ollama/mistral"],
                "require_structured": True,
                "temperature": 0.0,
            }
        }
    }


class BenchmarkRequest(BaseModel):
    """Request for running a benchmark."""

    dataset: str = Field(
        ...,
        description="Dataset name: 'arithmetic', 'algebra', or 'logic'",
    )
    models: list[str] | None = Field(
        default=None,
        description="List of models to benchmark. If None, uses all available.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of tasks to run",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset": "arithmetic",
                "models": ["ollama/llama3.2", "ollama/mistral"],
                "limit": 50,
            }
        }
    }
