"""Performance benchmarks for evaluation engine components."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List

from llm_eval_suite.core import EvalEngine
from llm_eval_suite.judge import MockJudgeBackend, build_judge_prompt, parse_judge_response
from llm_eval_suite.models import EvalSample
from llm_eval_suite.rubrics import build_helpfulness_rubric
from llm_eval_suite.storage import EvalStorage

logging.basicConfig(level=logging.WARNING)

NUM_ITERATIONS = 3


@dataclass
class BenchResult:
    """Benchmark result."""

    name: str
    mean_seconds: float
    items_processed: int
    throughput_per_sec: float


def bench_prompt_building() -> BenchResult:
    """Benchmark prompt construction speed."""
    rubric = build_helpfulness_rubric()
    sample = EvalSample(
        prompt="Explain quantum computing",
        response="Quantum computing uses qubits.",
        reference="Reference answer here.",
    )

    n_samples = 10000
    start = time.perf_counter()
    for _ in range(n_samples):
        build_judge_prompt(sample, rubric)
    elapsed = time.perf_counter() - start

    return BenchResult(
        name=f"Prompt Building ({n_samples} prompts)",
        mean_seconds=elapsed,
        items_processed=n_samples,
        throughput_per_sec=n_samples / elapsed,
    )


def bench_response_parsing() -> BenchResult:
    """Benchmark response parsing speed."""
    rubric = build_helpfulness_rubric()
    import json
    response = json.dumps({
        "scores": [
            {"criterion": c.name, "score": 4, "reasoning": "Test"}
            for c in rubric.criteria
        ]
    })

    n_samples = 10000
    start = time.perf_counter()
    for _ in range(n_samples):
        parse_judge_response(response, rubric)
    elapsed = time.perf_counter() - start

    return BenchResult(
        name=f"Response Parsing ({n_samples} responses)",
        mean_seconds=elapsed,
        items_processed=n_samples,
        throughput_per_sec=n_samples / elapsed,
    )


def bench_storage_writes() -> BenchResult:
    """Benchmark SQLite write performance."""
    from llm_eval_suite.models import CriterionScore, EvalResult

    storage = EvalStorage(db_path=":memory:")
    n_samples = 5000

    start = time.perf_counter()
    for i in range(n_samples):
        result = EvalResult(
            sample_id=f"bench-{i}",
            rubric_name="helpfulness",
            criterion_scores=[
                CriterionScore(criterion_name="accuracy", score=4.0, reasoning="Good"),
                CriterionScore(criterion_name="clarity", score=5.0, reasoning="Clear"),
            ],
            overall_score=4.5,
        )
        storage.save_result(result)
    elapsed = time.perf_counter() - start
    storage.close()

    return BenchResult(
        name=f"SQLite Writes ({n_samples} results)",
        mean_seconds=elapsed,
        items_processed=n_samples,
        throughput_per_sec=n_samples / elapsed,
    )


def bench_end_to_end() -> BenchResult:
    """Benchmark end-to-end evaluation pipeline."""
    n_samples = 100

    async def run_pipeline():
        judge = MockJudgeBackend(default_score=4)
        storage = EvalStorage(db_path=":memory:")
        engine = EvalEngine(judge=judge, storage=storage)
        rubric = build_helpfulness_rubric()

        samples = [
            EvalSample(
                prompt=f"Question {i}",
                response=f"Answer {i}",
                sample_id=f"e2e-{i}",
            )
            for i in range(n_samples)
        ]

        await engine.evaluate_batch(samples, rubric)
        storage.close()

    timings: List[float] = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        asyncio.run(run_pipeline())
        timings.append(time.perf_counter() - start)

    mean_time = sum(timings) / len(timings)
    return BenchResult(
        name=f"End-to-End ({n_samples} samples, {NUM_ITERATIONS} runs)",
        mean_seconds=mean_time,
        items_processed=n_samples,
        throughput_per_sec=n_samples / mean_time,
    )


def main() -> None:
    """Run all benchmarks."""
    benchmarks = [
        bench_prompt_building,
        bench_response_parsing,
        bench_storage_writes,
        bench_end_to_end,
    ]

    results: List[BenchResult] = []
    for bench_fn in benchmarks:
        results.append(bench_fn())

    header = f"{'Benchmark':<50} {'Time (s)':>10} {'Items':>8} {'Throughput':>14}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("LLM Eval Suite - Performance Benchmarks")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.name:<50} {r.mean_seconds:>10.4f} {r.items_processed:>8} "
            f"{r.throughput_per_sec:>12.2f}/s"
        )

    print(sep)
    print()


if __name__ == "__main__":
    main()
