"""Core evaluation engine orchestrating judge, rubrics, and storage."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llm_eval_suite.exceptions import EvalSuiteError
from llm_eval_suite.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeBackend,
    build_judge_prompt,
    compute_weighted_score,
    parse_judge_response,
)
from llm_eval_suite.models import EvalReport, EvalResult, EvalSample, Rubric
from llm_eval_suite.storage import EvalStorage

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "mock"


class EvalEngine:
    """Core evaluation engine.

    Orchestrates the evaluation pipeline:
    1. Accept samples and rubric
    2. Build judge prompts
    3. Dispatch to judge backend
    4. Parse and score responses
    5. Store results
    6. Generate reports
    """

    def __init__(
        self,
        judge: JudgeBackend,
        storage: Optional[EvalStorage] = None,
        judge_model_name: str = DEFAULT_JUDGE_MODEL,
    ) -> None:
        """Initialize evaluation engine.

        Args:
            judge: Backend for LLM-as-judge evaluation
            storage: Optional persistent storage for results
            judge_model_name: Name of the judge model for tracking
        """
        self.judge = judge
        self.storage = storage or EvalStorage()
        self.judge_model_name = judge_model_name
        logger.info(f"EvalEngine initialized with judge model: {judge_model_name}")

    async def evaluate_sample(
        self,
        sample: EvalSample,
        rubric: Rubric,
    ) -> EvalResult:
        """Evaluate a single sample against a rubric.

        Args:
            sample: The sample to evaluate
            rubric: The rubric to score against

        Returns:
            EvalResult with criterion scores and weighted overall score

        Raises:
            EvalSuiteError: If evaluation fails
        """
        logger.info(f"Evaluating sample {sample.sample_id} against rubric '{rubric.name}'")

        # Build the judge prompt
        user_prompt = build_judge_prompt(sample, rubric)

        # Get judge response
        try:
            raw_response = await self.judge.evaluate(JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            raise EvalSuiteError(f"Judge evaluation failed: {e}") from e

        # Parse scores
        criterion_scores = parse_judge_response(raw_response, rubric)

        # Compute weighted overall score
        overall_score = compute_weighted_score(criterion_scores, rubric)

        result = EvalResult(
            sample_id=sample.sample_id,
            rubric_name=rubric.name,
            criterion_scores=criterion_scores,
            overall_score=overall_score,
            judge_model=self.judge_model_name,
            metadata=sample.metadata,
        )

        # Persist result
        self.storage.save_result(result)

        logger.info(
            f"Sample {sample.sample_id}: overall_score={overall_score:.2f} "
            f"({len(criterion_scores)} criteria scored)"
        )
        return result

    async def evaluate_batch(
        self,
        samples: List[EvalSample],
        rubric: Rubric,
    ) -> List[EvalResult]:
        """Evaluate multiple samples against a rubric.

        Args:
            samples: List of samples to evaluate
            rubric: The rubric to score against

        Returns:
            List of EvalResult objects
        """
        logger.info(f"Evaluating batch of {len(samples)} samples against '{rubric.name}'")

        results: List[EvalResult] = []
        for i, sample in enumerate(samples):
            result = await self.evaluate_sample(sample, rubric)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(samples)} samples evaluated")

        logger.info(f"Batch complete: {len(results)} samples evaluated")
        return results

    def generate_report(self, rubric_name: str) -> EvalReport:
        """Generate an aggregated report for a rubric.

        Args:
            rubric_name: Name of the rubric to report on

        Returns:
            EvalReport with aggregated statistics
        """
        results = self.storage.get_results_by_rubric(rubric_name)

        if not results:
            return EvalReport(rubric_name=rubric_name)

        # Compute mean score
        scores = [r.overall_score for r in results]
        mean_score = sum(scores) / len(scores)

        # Score distribution (bucketed)
        distribution: Dict[str, int] = {}
        for score in scores:
            bucket = f"{int(score)}"
            distribution[bucket] = distribution.get(bucket, 0) + 1

        # Per-criterion means
        criterion_totals: Dict[str, List[float]] = {}
        for result in results:
            for cs in result.criterion_scores:
                if cs.criterion_name not in criterion_totals:
                    criterion_totals[cs.criterion_name] = []
                criterion_totals[cs.criterion_name].append(cs.score)

        criterion_means = {
            name: sum(vals) / len(vals)
            for name, vals in criterion_totals.items()
        }

        report = EvalReport(
            rubric_name=rubric_name,
            results=results,
            mean_score=mean_score,
            score_distribution=distribution,
            criterion_means=criterion_means,
        )

        logger.info(
            f"Report for '{rubric_name}': {len(results)} results, "
            f"mean={mean_score:.2f}"
        )
        return report
