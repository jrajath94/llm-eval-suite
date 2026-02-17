"""LLM Eval Suite: Evaluation framework with LLM-as-judge and custom rubrics."""

__version__ = "0.1.0"
__author__ = "Rajath John"

from llm_eval_suite.core import EvalEngine
from llm_eval_suite.judge import JudgeBackend, MockJudgeBackend
from llm_eval_suite.models import (
    Criterion,
    CriterionImportance,
    CriterionScore,
    EvalReport,
    EvalResult,
    EvalSample,
    Rubric,
)
from llm_eval_suite.rubrics import (
    get_rubric,
    initialize_default_rubrics,
    list_rubrics,
    register_rubric,
)
from llm_eval_suite.storage import EvalStorage

__all__ = [
    "EvalEngine",
    "JudgeBackend",
    "MockJudgeBackend",
    "EvalStorage",
    "Rubric",
    "Criterion",
    "CriterionImportance",
    "CriterionScore",
    "EvalSample",
    "EvalResult",
    "EvalReport",
    "register_rubric",
    "get_rubric",
    "list_rubrics",
    "initialize_default_rubrics",
]
