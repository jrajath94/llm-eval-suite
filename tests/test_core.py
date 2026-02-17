"""Tests for the core evaluation engine."""

from __future__ import annotations

import pytest

from llm_eval_suite.core import EvalEngine
from llm_eval_suite.judge import MockJudgeBackend
from llm_eval_suite.models import EvalSample
from llm_eval_suite.storage import EvalStorage


class TestEvalEngine:
    """Tests for EvalEngine orchestration."""

    @pytest.mark.asyncio
    async def test_evaluate_single_sample(self, sample_rubric, sample_input, mock_judge, memory_storage):
        """Test evaluating a single sample end-to-end."""
        engine = EvalEngine(
            judge=mock_judge,
            storage=memory_storage,
            judge_model_name="mock-4",
        )
        result = await engine.evaluate_sample(sample_input, sample_rubric)

        assert result.sample_id == "test-001"
        assert result.rubric_name == "test_rubric"
        assert result.overall_score > 0
        assert len(result.criterion_scores) > 0
        assert result.judge_model == "mock-4"

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, sample_rubric, mock_judge, memory_storage):
        """Test evaluating a batch of samples."""
        samples = [
            EvalSample(
                prompt=f"Question {i}",
                response=f"Answer {i}",
                sample_id=f"batch-{i}",
            )
            for i in range(5)
        ]

        engine = EvalEngine(judge=mock_judge, storage=memory_storage)
        results = await engine.evaluate_batch(samples, sample_rubric)

        assert len(results) == 5
        assert all(r.overall_score > 0 for r in results)

    @pytest.mark.asyncio
    async def test_results_persisted(self, sample_rubric, sample_input, mock_judge, memory_storage):
        """Test that results are saved to storage."""
        engine = EvalEngine(judge=mock_judge, storage=memory_storage)
        await engine.evaluate_sample(sample_input, sample_rubric)

        assert memory_storage.count_results() == 1
        stored = memory_storage.get_results_by_sample("test-001")
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_generate_report(self, sample_rubric, mock_judge, memory_storage):
        """Test report generation from stored results."""
        engine = EvalEngine(judge=mock_judge, storage=memory_storage)

        # Evaluate several samples
        for i in range(10):
            sample = EvalSample(
                prompt=f"Q{i}",
                response=f"A{i}",
                sample_id=f"report-{i}",
            )
            await engine.evaluate_sample(sample, sample_rubric)

        report = engine.generate_report("test_rubric")
        assert report.rubric_name == "test_rubric"
        assert len(report.results) == 10
        assert report.mean_score > 0
        assert len(report.criterion_means) > 0

    @pytest.mark.asyncio
    async def test_empty_report(self, mock_judge, memory_storage):
        """Test report for non-existent rubric."""
        engine = EvalEngine(judge=mock_judge, storage=memory_storage)
        report = engine.generate_report("nonexistent")
        assert len(report.results) == 0
        assert report.mean_score == 0.0

    @pytest.mark.asyncio
    async def test_different_judge_scores(self, sample_rubric, sample_input, memory_storage):
        """Test that different judge scores produce different overall scores."""
        judge_high = MockJudgeBackend(default_score=5)
        judge_low = MockJudgeBackend(default_score=1)

        engine_high = EvalEngine(judge=judge_high, storage=EvalStorage())
        engine_low = EvalEngine(judge=judge_low, storage=EvalStorage())

        result_high = await engine_high.evaluate_sample(sample_input, sample_rubric)
        result_low = await engine_low.evaluate_sample(sample_input, sample_rubric)

        assert result_high.overall_score > result_low.overall_score
