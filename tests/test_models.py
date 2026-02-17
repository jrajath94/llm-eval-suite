"""Tests for data models."""

from __future__ import annotations

import pytest

from llm_eval_suite.models import (
    Criterion,
    CriterionImportance,
    CriterionScore,
    EvalResult,
    EvalSample,
    Rubric,
)


class TestCriterion:
    """Tests for Criterion model."""

    def test_create_criterion(self):
        """Test basic criterion creation."""
        c = Criterion(name="accuracy", description="Is it correct?")
        assert c.name == "accuracy"
        assert c.weight == 1.0
        assert c.importance == CriterionImportance.IMPORTANT

    def test_negative_weight_raises(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Criterion(name="test", description="test", weight=-1.0)

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Criterion(name="", description="test")

    @pytest.mark.parametrize("importance", list(CriterionImportance))
    def test_importance_levels(self, importance):
        """Test all importance levels are valid."""
        c = Criterion(name="test", description="test", importance=importance)
        assert c.importance == importance


class TestRubric:
    """Tests for Rubric model."""

    def test_create_rubric(self):
        """Test basic rubric creation."""
        r = Rubric(name="test", description="Test rubric")
        assert r.name == "test"
        assert len(r.criteria) == 0
        assert r.scale_min == 1
        assert r.scale_max == 5

    def test_invalid_scale_raises(self):
        """Test that min >= max raises ValueError."""
        with pytest.raises(ValueError, match="scale_min"):
            Rubric(name="test", description="test", scale_min=5, scale_max=5)

    def test_add_criterion(self, sample_rubric):
        """Test adding criteria to rubric."""
        assert len(sample_rubric.criteria) == 3
        assert sample_rubric.total_weight == 6.0

    def test_total_weight(self):
        """Test total weight calculation."""
        r = Rubric(name="test", description="test")
        r.add_criterion(Criterion(name="a", description="a", weight=2.0))
        r.add_criterion(Criterion(name="b", description="b", weight=3.0))
        assert r.total_weight == 5.0


class TestEvalSample:
    """Tests for EvalSample model."""

    def test_auto_generated_id(self):
        """Test that sample_id is auto-generated when not provided."""
        s = EvalSample(prompt="test", response="test")
        assert len(s.sample_id) == 8

    def test_explicit_id(self):
        """Test setting explicit sample_id."""
        s = EvalSample(prompt="test", response="test", sample_id="my-id")
        assert s.sample_id == "my-id"

    def test_metadata(self):
        """Test metadata storage."""
        s = EvalSample(
            prompt="test",
            response="test",
            metadata={"model": "gpt-4", "temperature": 0.7},
        )
        assert s.metadata["model"] == "gpt-4"


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_auto_timestamp(self):
        """Test that timestamp is auto-generated."""
        r = EvalResult(sample_id="test", rubric_name="test")
        assert len(r.timestamp) > 0

    def test_criterion_scores(self):
        """Test result with criterion scores."""
        scores = [
            CriterionScore(criterion_name="accuracy", score=4.0, reasoning="Good"),
            CriterionScore(criterion_name="clarity", score=5.0, reasoning="Clear"),
        ]
        r = EvalResult(
            sample_id="test",
            rubric_name="test",
            criterion_scores=scores,
            overall_score=4.5,
        )
        assert len(r.criterion_scores) == 2
        assert r.overall_score == 4.5
