"""Structured output parsing for LLM responses."""

import json
import re
from typing import Any

from src.core.logging import get_logger
from src.models.math_task import MathSolution

logger = get_logger(__name__)


class ResponseParser:
    """Parser for extracting structured data from LLM responses."""

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response text.

        Handles various formats:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON mixed with other text

        Args:
            text: Raw response text from LLM.

        Returns:
            Parsed JSON dict or None if extraction failed.
        """
        # Try to parse as pure JSON first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        code_block_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object in text
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("json_extraction_failed", text_preview=text[:200])
        return None

    @staticmethod
    def extract_number(text: str) -> float | None:
        """Extract a number from text as fallback.

        Args:
            text: Text that may contain a number.

        Returns:
            Extracted number or None.
        """
        # Look for numbers in various formats
        patterns = [
            r"answer[:\s]+(-?\d+\.?\d*)",
            r"=\s*(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*)\s*$",
            r"(-?\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    @classmethod
    def parse_math_solution(cls, text: str) -> MathSolution | None:
        """Parse LLM response into MathSolution.

        Args:
            text: Raw response text from LLM.

        Returns:
            MathSolution or None if parsing failed.
        """
        # Try JSON extraction first
        json_data = cls.extract_json(text)

        if json_data:
            try:
                return MathSolution(
                    answer=float(json_data.get("answer", 0)),
                    steps=json_data.get("steps", []),
                    method=json_data.get("method", ""),
                    confidence=float(json_data.get("confidence", 1.0)),
                )
            except (ValueError, TypeError) as e:
                logger.warning("math_solution_parse_error", error=str(e))

        # Fallback: try to extract just the number
        number = cls.extract_number(text)
        if number is not None:
            return MathSolution(
                answer=number,
                steps=[],
                method="extracted from text",
                confidence=0.5,  # Lower confidence for fallback extraction
            )

        logger.warning("math_solution_extraction_failed", text_preview=text[:200])
        return None
