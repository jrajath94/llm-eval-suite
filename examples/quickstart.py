"""Quick start example: Evaluate LLM outputs with custom rubrics."""

from __future__ import annotations

import asyncio
import logging

from llm_eval_suite.core import EvalEngine
from llm_eval_suite.judge import MockJudgeBackend
from llm_eval_suite.models import EvalSample
from llm_eval_suite.rubrics import build_helpfulness_rubric, build_safety_rubric
from llm_eval_suite.storage import EvalStorage
from llm_eval_suite.utils import format_report_markdown, format_result_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main() -> None:
    """Run evaluation demonstration."""
    print("=" * 70)
    print("LLM Eval Suite - Quick Start Example")
    print("=" * 70)
    print()

    # Set up components
    judge = MockJudgeBackend(default_score=4)
    storage = EvalStorage(db_path=":memory:")
    engine = EvalEngine(judge=judge, storage=storage, judge_model_name="mock-v1")

    # Build rubrics
    helpfulness_rubric = build_helpfulness_rubric()
    safety_rubric = build_safety_rubric()

    # Sample data
    samples = [
        EvalSample(
            prompt="Explain quantum computing in simple terms.",
            response=(
                "Quantum computing uses quantum bits (qubits) that can represent "
                "0 and 1 simultaneously through superposition. This allows quantum "
                "computers to process many possibilities at once, making them powerful "
                "for specific problems like cryptography and drug discovery."
            ),
            reference="Quantum computing leverages quantum mechanics for computation.",
            sample_id="qc-001",
        ),
        EvalSample(
            prompt="What is the time complexity of binary search?",
            response="Binary search has O(log n) time complexity.",
            reference="O(log n)",
            sample_id="bs-001",
        ),
        EvalSample(
            prompt="Write a Python function to reverse a string.",
            response="def reverse(s): return s[::-1]",
            reference="def reverse_string(s): return s[::-1]",
            sample_id="rev-001",
        ),
    ]

    # Evaluate against helpfulness rubric
    print("-" * 70)
    print("Evaluating against 'helpfulness' rubric")
    print("-" * 70)
    print()

    results = await engine.evaluate_batch(samples, helpfulness_rubric)
    for result in results:
        print(format_result_summary(result))
        print()

    # Generate report
    report = engine.generate_report("helpfulness")
    print("-" * 70)
    print(format_report_markdown(report))

    # Evaluate against safety rubric
    print("-" * 70)
    print("Evaluating against 'safety' rubric")
    print("-" * 70)
    print()

    safety_results = await engine.evaluate_batch(samples, safety_rubric)
    safety_report = engine.generate_report("safety")
    print(format_report_markdown(safety_report))

    print(f"Total evaluations stored: {storage.count_results()}")
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

    storage.close()


if __name__ == "__main__":
    asyncio.run(main())
