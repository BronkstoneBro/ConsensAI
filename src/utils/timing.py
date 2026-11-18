"""Timing utilities for performance measurement."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    name: str
    elapsed_ms: float


@contextmanager
def measure_time(name: str, log: bool = True) -> Iterator[TimingResult]:
    """Context manager to measure execution time.

    Args:
        name: Name for the timing measurement.
        log: Whether to log the result.

    Yields:
        TimingResult with elapsed time after context exits.

    Example:
        with measure_time("model_query") as timing:
            result = await query_model()
        print(f"Took {timing.elapsed_ms}ms")
    """
    result = TimingResult(name=name, elapsed_ms=0.0)
    start = time.perf_counter()

    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000

        if log:
            logger.debug(
                "timing_measurement",
                name=name,
                elapsed_ms=round(result.elapsed_ms, 2),
            )
