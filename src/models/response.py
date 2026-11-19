"""Response models for API endpoints."""

from enum import Enum

from pydantic import BaseModel, Field


class AgreementLevel(str, Enum):
    """Level of agreement between models."""

    UNANIMOUS = "unanimous"  # All models agree
    MAJORITY = "majority"  # Most models agree
    SPLIT = "split"  # Models are evenly divided
    DISAGREEMENT = "disagreement"  # No clear consensus


class ModelResult(BaseModel):
    """Result from a single model."""

    model: str = Field(description="Model identifier")
    answer: float | str | None = Field(description="Model's answer")
    raw_response: str = Field(description="Raw text response from model")
    steps: list[str] | None = Field(default=None, description="Solution steps if provided")
    latency_ms: float = Field(description="Response time in milliseconds")
    tokens_prompt: int = Field(description="Prompt tokens used")
    tokens_completion: int = Field(description="Completion tokens used")
    error: str | None = Field(default=None, description="Error message if request failed")


class ConsensusResult(BaseModel):
    """Consensus analysis result."""

    answer: float | str | None = Field(description="Consensus answer")
    confidence: float = Field(ge=0, le=1, description="Confidence level (0.0 to 1.0)")
    agreement: AgreementLevel = Field(description="Level of agreement between models")
    correct: bool | None = Field(
        default=None,
        description="Whether answer is correct (only for tasks with known answer)",
    )


class ConsensusResponse(BaseModel):
    """Full response for consensus endpoint."""

    consensus: ConsensusResult = Field(description="Consensus analysis result")
    models: dict[str, ModelResult] = Field(description="Results from each model")
    metadata: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="Additional metadata (latency, method, judge info, etc.)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "consensus": {
                    "answer": 12.0,
                    "confidence": 1.0,
                    "agreement": "unanimous",
                    "correct": True,
                },
                "models": {
                    "ollama/llama3.2": {
                        "model": "ollama/llama3.2",
                        "answer": 12.0,
                        "raw_response": '{"answer": 12.0, "steps": ["15% = 0.15", "0.15 * 80 = 12"]}',
                        "steps": ["15% = 0.15", "0.15 * 80 = 12"],
                        "latency_ms": 234.5,
                        "tokens_prompt": 45,
                        "tokens_completion": 28,
                        "error": None,
                    }
                },
                "metadata": {
                    "total_latency_ms": 234.5,
                    "consensus_method": "exact_match",
                },
            }
        }
    }


class BenchmarkModelStats(BaseModel):
    """Statistics for a single model in benchmark."""

    accuracy: float = Field(ge=0, le=1, description="Fraction of correct answers")
    avg_latency_ms: float = Field(description="Average response time")
    avg_tokens: float = Field(description="Average tokens per response")
    total_tasks: int = Field(description="Total tasks attempted")
    correct: int = Field(description="Number of correct answers")
    incorrect: int = Field(description="Number of incorrect answers")
    errors: int = Field(default=0, description="Number of failed requests")


class ConsensusStats(BaseModel):
    """Statistics about consensus in benchmark."""

    unanimous_agreement: int = Field(description="Tasks where all models agreed")
    majority_agreement: int = Field(description="Tasks where majority agreed")
    disagreement: int = Field(description="Tasks with no clear consensus")


class BenchmarkResponse(BaseModel):
    """Response for benchmark endpoint."""

    benchmark_id: str = Field(description="Unique identifier for this benchmark run")
    status: str = Field(description="Benchmark status: 'running', 'completed', 'failed'")
    dataset: str = Field(description="Dataset used")
    results: dict[str, BenchmarkModelStats] = Field(
        description="Statistics per model"
    )
    consensus_stats: ConsensusStats = Field(
        description="Overall consensus statistics"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "benchmark_id": "bench_20241118_001",
                "status": "completed",
                "dataset": "arithmetic",
                "results": {
                    "ollama/llama3.2": {
                        "accuracy": 0.92,
                        "avg_latency_ms": 245.0,
                        "avg_tokens": 48.0,
                        "total_tasks": 100,
                        "correct": 92,
                        "incorrect": 8,
                        "errors": 0,
                    }
                },
                "consensus_stats": {
                    "unanimous_agreement": 85,
                    "majority_agreement": 12,
                    "disagreement": 3,
                },
            }
        }
    }


class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""

    total_requests: int = Field(description="Total consensus requests processed")
    models: dict[str, dict[str, float | int]] = Field(
        description="Per-model statistics"
    )
    consensus: dict[str, float] = Field(description="Consensus rate statistics")
