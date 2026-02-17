"""Custom exceptions for llm-eval-suite."""

from __future__ import annotations


class EvalSuiteError(Exception):
    """Base exception for all eval suite errors."""


class RubricValidationError(EvalSuiteError):
    """Raised when a rubric definition is invalid."""


class JudgeError(EvalSuiteError):
    """Raised when LLM judge fails to produce a valid evaluation."""


class StorageError(EvalSuiteError):
    """Raised when storage operations fail."""


class ConfigError(EvalSuiteError):
    """Raised when configuration is invalid."""
