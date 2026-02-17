"""Tests for SQLite storage."""

from __future__ import annotations

import pytest

from llm_eval_suite.models import CriterionScore, EvalResult
from llm_eval_suite.storage import EvalStorage


class TestEvalStorage:
    """Tests for EvalStorage SQLite backend."""

    def test_save_and_retrieve(self, memory_storage):
        """Test saving and retrieving a result."""
        result = EvalResult(
            sample_id="s-001",
            rubric_name="helpfulness",
            criterion_scores=[
                CriterionScore(criterion_name="accuracy", score=4.0, reasoning="Good"),
                CriterionScore(criterion_name="clarity", score=5.0, reasoning="Clear"),
            ],
            overall_score=4.5,
            judge_model="mock",
        )
        row_id = memory_storage.save_result(result)
        assert row_id > 0

        results = memory_storage.get_results_by_sample("s-001")
        assert len(results) == 1
        assert results[0].overall_score == 4.5
        assert len(results[0].criterion_scores) == 2

    def test_get_by_rubric(self, memory_storage):
        """Test filtering results by rubric name."""
        for i in range(5):
            memory_storage.save_result(EvalResult(
                sample_id=f"s-{i}",
                rubric_name="helpfulness",
                overall_score=4.0,
            ))
        for i in range(3):
            memory_storage.save_result(EvalResult(
                sample_id=f"s-safety-{i}",
                rubric_name="safety",
                overall_score=5.0,
            ))

        helpful_results = memory_storage.get_results_by_rubric("helpfulness")
        assert len(helpful_results) == 5

        safety_results = memory_storage.get_results_by_rubric("safety")
        assert len(safety_results) == 3

    def test_count_results(self, memory_storage):
        """Test counting results."""
        assert memory_storage.count_results() == 0

        memory_storage.save_result(EvalResult(
            sample_id="s-1",
            rubric_name="test",
            overall_score=3.0,
        ))
        assert memory_storage.count_results() == 1

    def test_get_all_results(self, memory_storage):
        """Test retrieving all results."""
        for i in range(10):
            memory_storage.save_result(EvalResult(
                sample_id=f"s-{i}",
                rubric_name="test",
                overall_score=float(i % 5 + 1),
            ))

        all_results = memory_storage.get_all_results()
        assert len(all_results) == 10

    def test_metadata_persistence(self, memory_storage):
        """Test that metadata is correctly persisted and retrieved."""
        result = EvalResult(
            sample_id="s-meta",
            rubric_name="test",
            overall_score=4.0,
            metadata={"model": "gpt-4", "temperature": 0.7},
        )
        memory_storage.save_result(result)

        retrieved = memory_storage.get_results_by_sample("s-meta")
        assert retrieved[0].metadata["model"] == "gpt-4"
        assert retrieved[0].metadata["temperature"] == 0.7

    def test_close_and_reopen(self):
        """Test closing and reopening storage."""
        storage = EvalStorage(db_path=":memory:")
        storage.save_result(EvalResult(
            sample_id="s-1",
            rubric_name="test",
            overall_score=3.0,
        ))
        storage.close()

        # After close, operations should fail
        from llm_eval_suite.exceptions import StorageError
        with pytest.raises(StorageError):
            storage.save_result(EvalResult(
                sample_id="s-2",
                rubric_name="test",
                overall_score=3.0,
            ))
