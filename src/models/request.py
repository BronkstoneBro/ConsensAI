"""Request models for API endpoints."""

from pydantic import BaseModel, Field


class ConsensusRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    models: list[str] | None = None
    require_structured: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_judge: bool = True


class BenchmarkRequest(BaseModel):
    dataset: str
    models: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
