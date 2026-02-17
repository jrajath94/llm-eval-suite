"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest

from llm_eval_suite.judge import MockJudgeBackend
from llm_eval_suite.models import (
    Criterion,
    CriterionImportance,
    EvalSample,
    Rubric,
)
from llm_eval_suite.storage import EvalStorage


@pytest.fixture
def sample_rubric() -> Rubric:
    """Fixture providing a test rubric with 3 criteria."""
    rubric = Rubric(name="test_rubric", description="Test rubric for unit tests")
    rubric.add_criterion(Criterion(
        name="accuracy",
        description="Is the answer correct?",
        importance=CriterionImportance.ESSENTIAL,
        weight=3.0,
        scoring_guide="5: Perfect. 1: Wrong.",
    ))
    rubric.add_criterion(Criterion(
        name="clarity",
        description="Is the answer clear?",
        importance=CriterionImportance.IMPORTANT,
        weight=2.0,
        scoring_guide="5: Crystal clear. 1: Confusing.",
    ))
    rubric.add_criterion(Criterion(
        name="brevity",
        description="Is the answer concise?",
        importance=CriterionImportance.OPTIONAL,
        weight=1.0,
        scoring_guide="5: Concise. 1: Verbose.",
    ))
    return rubric


@pytest.fixture
def sample_input() -> EvalSample:
    """Fixture providing a sample evaluation input."""
    return EvalSample(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris",
        sample_id="test-001",
    )


@pytest.fixture
def mock_judge() -> MockJudgeBackend:
    """Fixture providing a mock judge backend."""
    return MockJudgeBackend(default_score=4)


@pytest.fixture
def memory_storage() -> EvalStorage:
    """Fixture providing an in-memory storage instance."""
    storage = EvalStorage(db_path=":memory:")
    yield storage
    storage.close()
