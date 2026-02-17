"""Command-line interface for llm-eval-suite."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import click

from llm_eval_suite.core import EvalEngine
from llm_eval_suite.judge import MockJudgeBackend
from llm_eval_suite.models import EvalSample
from llm_eval_suite.rubrics import get_rubric, initialize_default_rubrics, list_rubrics
from llm_eval_suite.storage import EvalStorage
from llm_eval_suite.utils import format_report_markdown, format_result_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """LLM Eval Suite - Evaluate LLM outputs with custom rubrics."""
    initialize_default_rubrics()


@main.command()
def rubrics() -> None:
    """List all available rubrics."""
    names = list_rubrics()
    click.echo("Available rubrics:")
    for name in names:
        rubric = get_rubric(name)
        click.echo(f"  {name}: {rubric.description} ({len(rubric.criteria)} criteria)")


@main.command()
@click.option("--rubric", "-r", default="helpfulness", help="Rubric to use")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--response", "-o", required=True, help="LLM response to evaluate")
@click.option("--reference", default="", help="Optional reference answer")
@click.option("--db", default=":memory:", help="SQLite database path")
def evaluate(
    rubric: str,
    prompt: str,
    response: str,
    reference: str,
    db: str,
) -> None:
    """Evaluate a single prompt/response pair."""
    try:
        rubric_obj = get_rubric(rubric)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    sample = EvalSample(prompt=prompt, response=response, reference=reference)
    judge = MockJudgeBackend(default_score=4)
    storage = EvalStorage(db_path=db)
    engine = EvalEngine(judge=judge, storage=storage, judge_model_name="mock")

    result = asyncio.run(engine.evaluate_sample(sample, rubric_obj))
    click.echo(format_result_summary(result))
    storage.close()


@main.command()
@click.option("--rubric", "-r", default="helpfulness", help="Rubric to report on")
@click.option("--db", default=":memory:", help="SQLite database path")
def report(rubric: str, db: str) -> None:
    """Generate evaluation report."""
    storage = EvalStorage(db_path=db)
    judge = MockJudgeBackend()
    engine = EvalEngine(judge=judge, storage=storage)

    eval_report = engine.generate_report(rubric)
    click.echo(format_report_markdown(eval_report))
    storage.close()


if __name__ == "__main__":
    main()
