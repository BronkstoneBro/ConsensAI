"""Math task models for benchmarking."""

from enum import Enum

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Task difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    """Math task category."""

    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    LOGIC = "logic"
    GEOMETRY = "geometry"


class MathTask(BaseModel):
    """A mathematical task for benchmarking."""

    id: str = Field(description="Unique task identifier")
    question: str = Field(description="The math question in natural language")
    correct_answer: float = Field(description="The correct numerical answer")
    difficulty: Difficulty = Field(description="Task difficulty level")
    category: Category = Field(description="Math category")
    tags: list[str] = Field(default_factory=list, description="Additional tags")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "arith_001",
                "question": "Сколько будет 234 + 567?",
                "correct_answer": 801,
                "difficulty": "easy",
                "category": "arithmetic",
                "tags": ["addition", "integers"],
            }
        }
    }


class MathSolution(BaseModel):
    """Structured response from LLM for math task.

    This model is used with Pydantic AI to get structured outputs.
    """

    answer: float = Field(description="Numerical answer to the problem")
    steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step solution explanation",
    )
    method: str = Field(
        default="",
        description="Mathematical method or approach used",
    )
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Model's confidence in the answer (0.0 to 1.0)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": 12.0,
                "steps": [
                    "15% можно записать как 0.15",
                    "Умножаем: 0.15 × 80 = 12",
                ],
                "method": "процент от числа",
                "confidence": 0.95,
            }
        }
    }
