"""Math task models."""

from enum import Enum

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    LOGIC = "logic"
    GEOMETRY = "geometry"


class MathTask(BaseModel):
    id: str
    question: str
    correct_answer: float
    difficulty: Difficulty
    category: Category
    tags: list[str] = Field(default_factory=list)


class MathSolution(BaseModel):
    """Structured LLM response for math tasks."""

    answer: float
    steps: list[str] = Field(default_factory=list)
    method: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
