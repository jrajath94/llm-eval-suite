"""Tests for LLM-as-judge evaluation engine."""

from __future__ import annotations

import json

import pytest

from llm_eval_suite.exceptions import JudgeError
from llm_eval_suite.judge import (
    MockJudgeBackend,
    build_judge_prompt,
    compute_weighted_score,
    parse_judge_response,
)
from llm_eval_suite.models import (
    Criterion,
    CriterionImportance,
    CriterionScore,
    EvalSample,
    Rubric,
)


class TestBuildJudgePrompt:
    """Tests for judge prompt construction."""

    def test_prompt_contains_rubric_info(self, sample_rubric, sample_input):
        """Test that prompt includes rubric name and criteria."""
        prompt = build_judge_prompt(sample_input, sample_rubric)
        assert "test_rubric" in prompt
        assert "accuracy" in prompt
        assert "clarity" in prompt
        assert "brevity" in prompt

    def test_prompt_contains_sample(self, sample_rubric, sample_input):
        """Test that prompt includes the sample's prompt and response."""
        prompt = build_judge_prompt(sample_input, sample_rubric)
        assert "capital of France" in prompt
        assert "Paris" in prompt

    def test_prompt_includes_reference(self, sample_rubric, sample_input):
        """Test that reference answer is included when provided."""
        prompt = build_judge_prompt(sample_input, sample_rubric)
        assert "Reference Answer" in prompt

    def test_prompt_without_reference(self, sample_rubric):
        """Test prompt without reference answer."""
        sample = EvalSample(
            prompt="Hello",
            response="Hi there!",
        )
        prompt = build_judge_prompt(sample, sample_rubric)
        # Should not have reference section with content
        assert "Hello" in prompt
        assert "Hi there!" in prompt


class TestParseJudgeResponse:
    """Tests for parsing judge LLM responses."""

    def test_parse_valid_json(self, sample_rubric):
        """Test parsing well-formed JSON response."""
        response = json.dumps({
            "scores": [
                {"criterion": "accuracy", "score": 5, "reasoning": "Perfect"},
                {"criterion": "clarity", "score": 4, "reasoning": "Clear"},
                {"criterion": "brevity", "score": 3, "reasoning": "OK"},
            ]
        })
        scores = parse_judge_response(response, sample_rubric)
        assert len(scores) == 3
        assert scores[0].score == 5.0
        assert scores[0].criterion_name == "accuracy"

    def test_parse_json_in_code_block(self, sample_rubric):
        """Test parsing JSON embedded in markdown code block."""
        response = '```json\n{"scores": [{"criterion": "accuracy", "score": 4, "reasoning": "Good"}]}\n```'
        scores = parse_judge_response(response, sample_rubric)
        assert len(scores) == 1

    def test_parse_no_json_raises(self, sample_rubric):
        """Test that missing JSON raises JudgeError."""
        with pytest.raises(JudgeError, match="No JSON"):
            parse_judge_response("No JSON here", sample_rubric)

    def test_parse_invalid_json_raises(self, sample_rubric):
        """Test that invalid JSON raises JudgeError."""
        with pytest.raises(JudgeError, match="Invalid JSON"):
            parse_judge_response('{"scores": [invalid]}', sample_rubric)

    def test_clamp_score_to_range(self, sample_rubric):
        """Test that out-of-range scores are clamped."""
        response = json.dumps({
            "scores": [
                {"criterion": "accuracy", "score": 10, "reasoning": "Over"},
            ]
        })
        scores = parse_judge_response(response, sample_rubric)
        assert scores[0].score == 5.0  # Clamped to scale_max

    def test_unknown_criterion_skipped(self, sample_rubric):
        """Test that unknown criteria are skipped with warning."""
        response = json.dumps({
            "scores": [
                {"criterion": "accuracy", "score": 4, "reasoning": "OK"},
                {"criterion": "unknown_crit", "score": 3, "reasoning": "Skip"},
            ]
        })
        scores = parse_judge_response(response, sample_rubric)
        assert len(scores) == 1
        assert scores[0].criterion_name == "accuracy"


class TestComputeWeightedScore:
    """Tests for weighted score computation."""

    def test_equal_weights(self):
        """Test scoring with equal weights."""
        rubric = Rubric(name="test", description="test")
        rubric.add_criterion(Criterion(name="a", description="a", weight=1.0))
        rubric.add_criterion(Criterion(name="b", description="b", weight=1.0))

        scores = [
            CriterionScore(criterion_name="a", score=4.0),
            CriterionScore(criterion_name="b", score=2.0),
        ]
        result = compute_weighted_score(scores, rubric)
        assert result == 3.0  # (4*1 + 2*1) / (1+1)

    def test_unequal_weights(self):
        """Test scoring with different weights."""
        rubric = Rubric(name="test", description="test")
        rubric.add_criterion(Criterion(name="a", description="a", weight=3.0))
        rubric.add_criterion(Criterion(name="b", description="b", weight=1.0))

        scores = [
            CriterionScore(criterion_name="a", score=5.0),
            CriterionScore(criterion_name="b", score=1.0),
        ]
        result = compute_weighted_score(scores, rubric)
        assert result == 4.0  # (5*3 + 1*1) / (3+1)

    def test_empty_scores(self):
        """Test with no scores returns 0."""
        rubric = Rubric(name="test", description="test")
        result = compute_weighted_score([], rubric)
        assert result == 0.0


class TestMockJudgeBackend:
    """Tests for mock judge implementation."""

    @pytest.mark.asyncio
    async def test_mock_returns_json(self, mock_judge, sample_rubric, sample_input):
        """Test mock judge returns valid JSON with scores."""
        prompt = build_judge_prompt(sample_input, sample_rubric)
        response = await mock_judge.evaluate("system", prompt)

        data = json.loads(response)
        assert "scores" in data
        assert len(data["scores"]) > 0

    @pytest.mark.asyncio
    async def test_mock_increments_call_count(self, mock_judge):
        """Test that call counter increments."""
        assert mock_judge.call_count == 0
        await mock_judge.evaluate("system", "- accuracy: test")
        assert mock_judge.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_default_score(self):
        """Test configurable default score."""
        judge = MockJudgeBackend(default_score=3)
        response = await judge.evaluate("system", "- accuracy: test\n- clarity: test")
        data = json.loads(response)
        for s in data["scores"]:
            assert s["score"] == 3
