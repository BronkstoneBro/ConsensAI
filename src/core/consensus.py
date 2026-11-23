"""Consensus analysis algorithms."""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

from src.core.logging import get_logger
from src.models.math_task import MathSolution
from src.models.response import AgreementLevel

logger = get_logger(__name__)


@dataclass
class ConsensusAnalysisResult:
    answer: float | None
    confidence: float
    agreement: AgreementLevel
    method: str
    vote_distribution: dict[float, int]


class IConsensusStrategy(ABC):
    @abstractmethod
    def analyze(self, solutions: dict[str, MathSolution]) -> ConsensusAnalysisResult:
        pass


class ExactMatchStrategy(IConsensusStrategy):
    """Groups answers within tolerance and returns the most common."""

    def __init__(self, tolerance: float = 0.001) -> None:
        self._tolerance = tolerance

    def analyze(self, solutions: dict[str, MathSolution]) -> ConsensusAnalysisResult:
        if not solutions:
            return ConsensusAnalysisResult(
                answer=None,
                confidence=0.0,
                agreement=AgreementLevel.DISAGREEMENT,
                method="exact_match",
                vote_distribution={},
            )

        # Group answers by value (within tolerance)
        answer_groups: dict[float, list[str]] = {}

        for model, solution in solutions.items():
            answer = solution.answer
            # Find existing group within tolerance
            found_group = False
            for group_answer in answer_groups:
                if abs(answer - group_answer) <= self._tolerance:
                    answer_groups[group_answer].append(model)
                    found_group = True
                    break

            if not found_group:
                answer_groups[answer] = [model]

        # Find the answer with most votes
        total_models = len(solutions)
        vote_distribution = {
            answer: len(models)
            for answer, models in answer_groups.items()
        }

        winning_answer = max(vote_distribution, key=vote_distribution.get)  # type: ignore
        winning_votes = vote_distribution[winning_answer]

        # Determine agreement level
        if winning_votes == total_models:
            agreement = AgreementLevel.UNANIMOUS
            confidence = 1.0
        elif winning_votes > total_models / 2:
            agreement = AgreementLevel.MAJORITY
            confidence = winning_votes / total_models
        elif winning_votes == total_models / 2 and total_models > 1:
            agreement = AgreementLevel.SPLIT
            confidence = 0.5
        else:
            agreement = AgreementLevel.DISAGREEMENT
            confidence = winning_votes / total_models

        logger.info(
            "exact_match_consensus",
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement.value,
            vote_distribution=vote_distribution,
        )

        return ConsensusAnalysisResult(
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement,
            method="exact_match",
            vote_distribution=vote_distribution,
        )


class MajorityVoteStrategy(IConsensusStrategy):
    """Rounds answers to precision and returns the most common."""

    def __init__(self, precision: int = 2) -> None:
        self._precision = precision

    def analyze(self, solutions: dict[str, MathSolution]) -> ConsensusAnalysisResult:
        if not solutions:
            return ConsensusAnalysisResult(
                answer=None,
                confidence=0.0,
                agreement=AgreementLevel.DISAGREEMENT,
                method="majority_vote",
                vote_distribution={},
            )

        # Round answers and count votes
        rounded_answers = {
            model: round(solution.answer, self._precision)
            for model, solution in solutions.items()
        }

        vote_counts = Counter(rounded_answers.values())
        total_models = len(solutions)

        # Get winner
        winning_answer, winning_votes = vote_counts.most_common(1)[0]

        # Determine agreement level
        if winning_votes == total_models:
            agreement = AgreementLevel.UNANIMOUS
            confidence = 1.0
        elif winning_votes > total_models / 2:
            agreement = AgreementLevel.MAJORITY
            confidence = winning_votes / total_models
        elif winning_votes == total_models / 2 and total_models > 1:
            agreement = AgreementLevel.SPLIT
            confidence = 0.5
        else:
            agreement = AgreementLevel.DISAGREEMENT
            confidence = winning_votes / total_models

        vote_distribution = dict(vote_counts)

        logger.info(
            "majority_vote_consensus",
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement.value,
            vote_distribution=vote_distribution,
        )

        return ConsensusAnalysisResult(
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement,
            method="majority_vote",
            vote_distribution=vote_distribution,
        )


class WeightedConfidenceStrategy(IConsensusStrategy):
    """Weights votes by model-reported confidence."""

    def analyze(self, solutions: dict[str, MathSolution]) -> ConsensusAnalysisResult:
        if not solutions:
            return ConsensusAnalysisResult(
                answer=None,
                confidence=0.0,
                agreement=AgreementLevel.DISAGREEMENT,
                method="weighted_confidence",
                vote_distribution={},
            )

        # Group answers and sum their confidence weights
        answer_weights: dict[float, float] = {}
        answer_counts: dict[float, int] = {}

        for model, solution in solutions.items():
            answer = round(solution.answer, 2)
            weight = solution.confidence

            if answer in answer_weights:
                answer_weights[answer] += weight
                answer_counts[answer] += 1
            else:
                answer_weights[answer] = weight
                answer_counts[answer] = 1

        # Find answer with highest total weight
        winning_answer = max(answer_weights, key=answer_weights.get)  # type: ignore
        total_weight = sum(answer_weights.values())
        winning_weight = answer_weights[winning_answer]

        # Normalize confidence
        confidence = winning_weight / total_weight if total_weight > 0 else 0.0

        # Determine agreement based on count
        total_models = len(solutions)
        winning_votes = answer_counts[winning_answer]

        if winning_votes == total_models:
            agreement = AgreementLevel.UNANIMOUS
        elif winning_votes > total_models / 2:
            agreement = AgreementLevel.MAJORITY
        elif winning_votes == total_models / 2 and total_models > 1:
            agreement = AgreementLevel.SPLIT
        else:
            agreement = AgreementLevel.DISAGREEMENT

        logger.info(
            "weighted_confidence_consensus",
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement.value,
        )

        return ConsensusAnalysisResult(
            answer=winning_answer,
            confidence=confidence,
            agreement=agreement,
            method="weighted_confidence",
            vote_distribution=answer_counts,
        )


class ConsensusAnalyzer:
    """Unified interface to different consensus algorithms."""

    def __init__(self) -> None:
        self._strategies: dict[str, IConsensusStrategy] = {
            "exact_match": ExactMatchStrategy(),
            "majority_vote": MajorityVoteStrategy(),
            "weighted_confidence": WeightedConfidenceStrategy(),
        }

    def analyze(
        self,
        solutions: dict[str, MathSolution],
        strategy: str = "exact_match",
    ) -> ConsensusAnalysisResult:
        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(self._strategies.keys())}"
            )
        return self._strategies[strategy].analyze(solutions)

    def auto_analyze(self, solutions: dict[str, MathSolution]) -> ConsensusAnalysisResult:
        """Uses exact_match, falls back to weighted_confidence on disagreement."""
        # Try exact match first
        result = self._strategies["exact_match"].analyze(solutions)

        # If no clear consensus, try weighted confidence
        if result.agreement == AgreementLevel.DISAGREEMENT:
            logger.info("auto_analyze_fallback_to_weighted")
            result = self._strategies["weighted_confidence"].analyze(solutions)

        return result
