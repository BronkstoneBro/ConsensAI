"""Response models for API endpoints."""

from enum import Enum

from pydantic import BaseModel, Field


class AgreementLevel(str, Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SPLIT = "split"
    DISAGREEMENT = "disagreement"


class ModelResult(BaseModel):
    model: str
    answer: float | str | None
    raw_response: str
    steps: list[str] | None = None
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    error: str | None = None


class ConsensusResult(BaseModel):
    answer: float | str | None
    confidence: float = Field(ge=0, le=1)
    agreement: AgreementLevel
    correct: bool | None = None


class ConsensusResponse(BaseModel):
    consensus: ConsensusResult
    models: dict[str, ModelResult]
    metadata: dict[str, float | int | str | bool] = Field(default_factory=dict)


class BenchmarkModelStats(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    avg_latency_ms: float
    avg_tokens: float
    total_tasks: int
    correct: int
    incorrect: int
    errors: int = 0


class ConsensusStats(BaseModel):
    unanimous_agreement: int
    majority_agreement: int
    disagreement: int


class BenchmarkResponse(BaseModel):
    benchmark_id: str
    status: str
    dataset: str
    results: dict[str, BenchmarkModelStats]
    consensus_stats: ConsensusStats


class MetricsResponse(BaseModel):
    total_requests: int
    models: dict[str, dict[str, float | int]]
    consensus: dict[str, float]
