"""Dataset loading for benchmarking."""

import json
from pathlib import Path

from src.core.exceptions import BenchmarkError
from src.core.logging import get_logger
from src.models.math_task import Category, Difficulty, MathTask

logger = get_logger(__name__)

DATASETS_PATH = Path(__file__).parent.parent.parent / "data" / "math_tasks"
AVAILABLE_DATASETS = ["arithmetic", "algebra", "logic"]


def load_dataset(name: str, limit: int | None = None) -> list[MathTask]:
    if name not in AVAILABLE_DATASETS:
        raise BenchmarkError(
            f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}"
        )

    file_path = DATASETS_PATH / f"{name}.json"

    if not file_path.exists():
        raise BenchmarkError(f"Dataset file not found: {file_path}")

    logger.info("loading_dataset", name=name, path=str(file_path))

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise BenchmarkError(f"Invalid JSON in dataset {name}: {e}") from e

    difficulty_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }

    dataset_difficulty = difficulty_map.get(data.get("difficulty", "easy"), Difficulty.EASY)

    tasks = []
    for task_data in data.get("tasks", []):
        task = MathTask(
            id=task_data["id"],
            question=task_data["question"],
            correct_answer=float(task_data["answer"]),
            difficulty=dataset_difficulty,
            category=_map_category(task_data.get("category", "arithmetic")),
            tags=[task_data.get("category", "")],
        )
        tasks.append(task)

        if limit and len(tasks) >= limit:
            break

    logger.info("dataset_loaded", name=name, task_count=len(tasks))
    return tasks


def _map_category(category_str: str) -> Category:
    category_mapping = {
        "addition": Category.ARITHMETIC,
        "subtraction": Category.ARITHMETIC,
        "multiplication": Category.ARITHMETIC,
        "division": Category.ARITHMETIC,
        "percentage": Category.ARITHMETIC,
        "fractions": Category.ARITHMETIC,
        "exponents": Category.ARITHMETIC,
        "roots": Category.ARITHMETIC,
        "linear": Category.ALGEBRA,
        "quadratic": Category.ALGEBRA,
        "system": Category.ALGEBRA,
        "word_problem": Category.LOGIC,
        "rate": Category.LOGIC,
        "discount": Category.LOGIC,
        "work": Category.LOGIC,
        "markup": Category.LOGIC,
        "average": Category.LOGIC,
        "age": Category.LOGIC,
        "ratio": Category.LOGIC,
        "sequence": Category.LOGIC,
        "geometry": Category.GEOMETRY,
        "interest": Category.LOGIC,
    }
    return category_mapping.get(category_str, Category.ARITHMETIC)


def get_all_tasks(limit_per_dataset: int | None = None) -> list[MathTask]:
    all_tasks = []
    for dataset_name in AVAILABLE_DATASETS:
        tasks = load_dataset(dataset_name, limit=limit_per_dataset)
        all_tasks.extend(tasks)
    return all_tasks
