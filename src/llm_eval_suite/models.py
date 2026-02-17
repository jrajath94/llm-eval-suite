"""Data models for evaluation framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CriterionImportance(Enum):
    """Importance level for rubric criteria."""

    ESSENTIAL = "essential"
    IMPORTANT = "important"
    OPTIONAL = "optional"


@dataclass
class Criterion:
    """A single evaluation criterion within a rubric.

    Args:
        name: Short identifier for the criterion
        description: What this criterion measures
        importance: How critical this criterion is to overall quality
        weight: Numeric weight for scoring (higher = more impact)
        scoring_guide: Instructions for the judge on how to score
    """

    name: str
    description: str
    importance: CriterionImportance = CriterionImportance.IMPORTANT
    weight: float = 1.0
    scoring_guide: str = ""

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if not self.name.strip():
            raise ValueError("Criterion name cannot be empty")


@dataclass
class Rubric:
    """A collection of criteria for evaluating LLM output.

    Args:
        name: Human-readable rubric name
        description: What this rubric evaluates
        criteria: List of evaluation criteria
        scale_min: Minimum score on the rating scale
        scale_max: Maximum score on the rating scale
    """

    name: str
    description: str
    criteria: List[Criterion] = field(default_factory=list)
    scale_min: int = 1
    scale_max: int = 5

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"scale_min ({self.scale_min}) must be less than "
                f"scale_max ({self.scale_max})"
            )

    @property
    def total_weight(self) -> float:
        """Sum of all criterion weights."""
        return sum(c.weight for c in self.criteria)

    def add_criterion(self, criterion: Criterion) -> None:
        """Add a criterion to the rubric.

        Args:
            criterion: Criterion to add
        """
        self.criteria.append(criterion)


@dataclass
class EvalSample:
    """A single input/output pair to evaluate.

    Args:
        sample_id: Unique identifier
        prompt: The input prompt
        response: The LLM-generated response
        reference: Optional reference/expected answer
        metadata: Additional context
    """

    prompt: str
    response: str
    sample_id: str = ""
    reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            self.sample_id = str(uuid.uuid4())[:8]


@dataclass
class CriterionScore:
    """Score for a single criterion.

    Args:
        criterion_name: Name of the criterion scored
        score: Numeric score
        reasoning: Judge's explanation for the score
    """

    criterion_name: str
    score: float
    reasoning: str = ""


@dataclass
class EvalResult:
    """Complete evaluation result for a single sample.

    Args:
        sample_id: ID of the evaluated sample
        rubric_name: Name of the rubric used
        criterion_scores: Individual criterion scores
        overall_score: Weighted average score
        judge_model: Model used for judging
        timestamp: When evaluation was performed
        metadata: Additional eval metadata
    """

    sample_id: str
    rubric_name: str
    criterion_scores: List[CriterionScore] = field(default_factory=list)
    overall_score: float = 0.0
    judge_model: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class EvalReport:
    """Aggregated report across multiple evaluations.

    Args:
        rubric_name: Rubric used for all evaluations
        results: Individual eval results
        mean_score: Average overall score
        score_distribution: Histogram of scores
        criterion_means: Per-criterion average scores
    """

    rubric_name: str
    results: List[EvalResult] = field(default_factory=list)
    mean_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    criterion_means: Dict[str, float] = field(default_factory=dict)
