"""LLM-as-Judge evaluation engine."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from llm_eval_suite.exceptions import JudgeError
from llm_eval_suite.models import (
    Criterion,
    CriterionScore,
    EvalResult,
    EvalSample,
    Rubric,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Score the given response according to "
    "the rubric criteria. For each criterion, provide a score and brief reasoning. "
    "Respond ONLY with valid JSON."
)

MAX_SCORE_RETRIES = 2


def build_judge_prompt(
    sample: EvalSample,
    rubric: Rubric,
) -> str:
    """Build the evaluation prompt for the judge LLM.

    Args:
        sample: The sample to evaluate
        rubric: The rubric to score against

    Returns:
        Formatted prompt string
    """
    criteria_text = ""
    for criterion in rubric.criteria:
        criteria_text += (
            f"- {criterion.name}: {criterion.description}\n"
            f"  Scoring guide: {criterion.scoring_guide}\n"
        )

    reference_section = ""
    if sample.reference:
        reference_section = f"\n## Reference Answer\n{sample.reference}\n"

    prompt = (
        f"## Rubric: {rubric.name}\n"
        f"{rubric.description}\n\n"
        f"Score range: {rubric.scale_min} to {rubric.scale_max}\n\n"
        f"## Criteria\n{criteria_text}\n"
        f"## Prompt\n{sample.prompt}\n\n"
        f"## Response to Evaluate\n{sample.response}\n"
        f"{reference_section}\n"
        f"## Instructions\n"
        f"Score each criterion from {rubric.scale_min} to {rubric.scale_max}. "
        f"Return JSON with this exact structure:\n"
        f'{{"scores": [{{"criterion": "<name>", "score": <int>, '
        f'"reasoning": "<brief explanation>"}}]}}'
    )
    return prompt


def parse_judge_response(
    raw_response: str,
    rubric: Rubric,
) -> list[CriterionScore]:
    """Parse the judge LLM's JSON response into criterion scores.

    Args:
        raw_response: Raw text from the judge LLM
        rubric: The rubric to validate scores against

    Returns:
        List of CriterionScore objects

    Raises:
        JudgeError: If response cannot be parsed
    """
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[\s\S]*\}', raw_response)
    if not json_match:
        raise JudgeError(f"No JSON found in judge response: {raw_response[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise JudgeError(f"Invalid JSON in judge response: {e}") from e

    scores_data = data.get("scores", [])
    if not scores_data:
        raise JudgeError("No 'scores' key found in judge response")

    criterion_names = {c.name for c in rubric.criteria}
    criterion_scores: list[CriterionScore] = []

    for score_entry in scores_data:
        name = score_entry.get("criterion", "")
        score_val = score_entry.get("score", 0)
        reasoning = score_entry.get("reasoning", "")

        if name not in criterion_names:
            logger.warning(f"Unknown criterion '{name}' in judge response, skipping")
            continue

        # Clamp score to rubric range
        score_val = max(rubric.scale_min, min(rubric.scale_max, int(score_val)))

        criterion_scores.append(CriterionScore(
            criterion_name=name,
            score=float(score_val),
            reasoning=reasoning,
        ))

    if not criterion_scores:
        raise JudgeError("No valid criterion scores parsed from judge response")

    return criterion_scores


def compute_weighted_score(
    criterion_scores: list[CriterionScore],
    rubric: Rubric,
) -> float:
    """Compute weighted average score from criterion scores.

    Args:
        criterion_scores: Individual criterion scores
        rubric: The rubric with criterion weights

    Returns:
        Weighted average score
    """
    criteria_by_name = {c.name: c for c in rubric.criteria}
    total_weight = 0.0
    weighted_sum = 0.0

    for cs in criterion_scores:
        criterion = criteria_by_name.get(cs.criterion_name)
        if criterion:
            weighted_sum += cs.score * criterion.weight
            total_weight += criterion.weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


class JudgeBackend(ABC):
    """Abstract interface for judge LLM backends."""

    @abstractmethod
    async def evaluate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the judge LLM and get a response.

        Args:
            system_prompt: System-level instruction
            user_prompt: The evaluation prompt

        Returns:
            Raw text response from the judge
        """
        ...


class MockJudgeBackend(JudgeBackend):
    """Mock judge backend that returns deterministic scores for testing."""

    def __init__(self, default_score: int = 4) -> None:
        """Initialize mock judge.

        Args:
            default_score: Score to assign to all criteria
        """
        self.default_score = default_score
        self.call_count = 0

    async def evaluate(self, system_prompt: str, user_prompt: str) -> str:
        """Return deterministic JSON scores.

        Args:
            system_prompt: Ignored
            user_prompt: Parsed to extract criterion names

        Returns:
            JSON string with scores for all criteria found in prompt
        """
        self.call_count += 1

        # Extract criterion names from prompt
        criteria_names = re.findall(r'- (\w+):', user_prompt)
        scores = [
            {
                "criterion": name,
                "score": self.default_score,
                "reasoning": f"Mock evaluation: score {self.default_score} for {name}",
            }
            for name in criteria_names
        ]
        return json.dumps({"scores": scores})


class HttpJudgeBackend(JudgeBackend):
    """Judge backend that calls an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4",
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize HTTP judge backend.

        Args:
            base_url: API base URL
            api_key: API authentication key
            model: Model identifier
            timeout_seconds: Request timeout
        """
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def evaluate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the judge LLM API.

        Args:
            system_prompt: System instruction
            user_prompt: Evaluation prompt

        Returns:
            Raw text response

        Raises:
            JudgeError: If API call fails
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise JudgeError(f"Judge API call failed: {e}") from e
