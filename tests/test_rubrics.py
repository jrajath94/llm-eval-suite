"""Tests for rubric registry and built-in rubrics."""

from __future__ import annotations

import pytest

from llm_eval_suite.exceptions import RubricValidationError
from llm_eval_suite.models import Criterion, Rubric
from llm_eval_suite.rubrics import (
    _RUBRIC_REGISTRY,
    build_code_quality_rubric,
    build_helpfulness_rubric,
    build_safety_rubric,
    get_rubric,
    initialize_default_rubrics,
    list_rubrics,
    register_rubric,
)


class TestRubricRegistry:
    """Tests for rubric registration and retrieval."""

    def setup_method(self):
        """Clear registry before each test."""
        _RUBRIC_REGISTRY.clear()

    def test_register_and_get(self, sample_rubric):
        """Test registering and retrieving a rubric."""
        register_rubric(sample_rubric)
        retrieved = get_rubric("test_rubric")
        assert retrieved.name == "test_rubric"
        assert len(retrieved.criteria) == 3

    def test_register_empty_rubric_raises(self):
        """Test that registering a rubric with no criteria raises error."""
        empty_rubric = Rubric(name="empty", description="Empty rubric")
        with pytest.raises(RubricValidationError):
            register_rubric(empty_rubric)

    def test_get_nonexistent_raises(self):
        """Test that getting a missing rubric raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_rubric("nonexistent")

    def test_list_rubrics(self, sample_rubric):
        """Test listing registered rubric names."""
        register_rubric(sample_rubric)
        names = list_rubrics()
        assert "test_rubric" in names


class TestBuiltInRubrics:
    """Tests for built-in rubric templates."""

    def setup_method(self):
        """Clear registry before each test."""
        _RUBRIC_REGISTRY.clear()

    def test_helpfulness_rubric(self):
        """Test helpfulness rubric has expected criteria."""
        rubric = build_helpfulness_rubric()
        assert rubric.name == "helpfulness"
        assert len(rubric.criteria) == 5
        criterion_names = {c.name for c in rubric.criteria}
        assert "relevance" in criterion_names
        assert "accuracy" in criterion_names

    def test_safety_rubric(self):
        """Test safety rubric has expected criteria."""
        rubric = build_safety_rubric()
        assert rubric.name == "safety"
        assert len(rubric.criteria) == 3
        criterion_names = {c.name for c in rubric.criteria}
        assert "harmlessness" in criterion_names

    def test_code_quality_rubric(self):
        """Test code quality rubric has expected criteria."""
        rubric = build_code_quality_rubric()
        assert rubric.name == "code_quality"
        assert len(rubric.criteria) == 4
        criterion_names = {c.name for c in rubric.criteria}
        assert "correctness" in criterion_names

    def test_initialize_defaults(self):
        """Test initializing all default rubrics."""
        initialize_default_rubrics()
        names = list_rubrics()
        assert len(names) == 3
        assert "helpfulness" in names
        assert "safety" in names
        assert "code_quality" in names

    @pytest.mark.parametrize("builder,expected_name", [
        (build_helpfulness_rubric, "helpfulness"),
        (build_safety_rubric, "safety"),
        (build_code_quality_rubric, "code_quality"),
    ])
    def test_rubric_weights_positive(self, builder, expected_name):
        """Test that all rubric criteria have positive weights."""
        rubric = builder()
        assert rubric.name == expected_name
        for criterion in rubric.criteria:
            assert criterion.weight > 0
