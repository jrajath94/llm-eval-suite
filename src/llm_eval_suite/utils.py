"""Utility functions for llm-eval-suite."""

from __future__ import annotations

import logging
from typing import List

from llm_eval_suite.models import EvalReport, EvalResult

logger = logging.getLogger(__name__)


def format_report_markdown(report: EvalReport) -> str:
    """Format an evaluation report as markdown.

    Args:
        report: EvalReport to format

    Returns:
        Markdown-formatted report string
    """
    lines: List[str] = []
    lines.append(f"# Evaluation Report: {report.rubric_name}")
    lines.append(f"**Samples evaluated:** {len(report.results)}")
    lines.append(f"**Mean score:** {report.mean_score:.2f}")
    lines.append("")

    if report.criterion_means:
        lines.append("## Per-Criterion Averages")
        lines.append("| Criterion | Mean Score |")
        lines.append("|-----------|------------|")
        for name, mean in sorted(report.criterion_means.items()):
            lines.append(f"| {name} | {mean:.2f} |")
        lines.append("")

    if report.score_distribution:
        lines.append("## Score Distribution")
        lines.append("| Score | Count |")
        lines.append("|-------|-------|")
        for bucket, count in sorted(report.score_distribution.items()):
            lines.append(f"| {bucket} | {count} |")
        lines.append("")

    return "\n".join(lines)


def format_result_summary(result: EvalResult) -> str:
    """Format a single evaluation result as a summary string.

    Args:
        result: EvalResult to format

    Returns:
        Human-readable summary string
    """
    lines: List[str] = []
    lines.append(f"Sample: {result.sample_id} | Rubric: {result.rubric_name}")
    lines.append(f"Overall Score: {result.overall_score:.2f}")
    for cs in result.criterion_scores:
        lines.append(f"  {cs.criterion_name}: {cs.score:.1f} - {cs.reasoning}")
    return "\n".join(lines)
