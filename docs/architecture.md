# Architecture: llm-eval-suite

## Overview

llm-eval-suite implements a modular evaluation pipeline for assessing LLM output quality using the **LLM-as-judge** paradigm. The architecture separates concerns into five layers: data models, rubric management, judge orchestration, persistence, and reporting.

## Component Responsibilities

### Models (`models.py`)

- **Criterion**: Single evaluation dimension with name, description, weight, importance level, and scoring guide
- **Rubric**: Collection of weighted criteria with a scoring scale (default 1-5)
- **EvalSample**: Input to the pipeline -- prompt/response pair with optional reference answer
- **CriterionScore**: Output per criterion -- numeric score with judge reasoning
- **EvalResult**: Complete evaluation result for one sample -- criterion scores + weighted overall
- **EvalReport**: Aggregated statistics across multiple evaluations -- means, distributions

All models use `dataclasses` with `__post_init__` validation.

### Rubrics (`rubrics.py`)

- Global registry pattern using a module-level dict `_RUBRIC_REGISTRY`
- Three built-in templates: `helpfulness` (5 criteria), `safety` (3 criteria), `code_quality` (4 criteria)
- Extensible via `register_rubric()` for custom evaluation dimensions
- Each criterion has a `weight` that controls its influence on the overall score

### Judge (`judge.py`)

- **`build_judge_prompt()`**: Constructs a structured prompt with rubric criteria, scoring guides, the sample, and JSON output format instructions
- **`parse_judge_response()`**: Extracts JSON from raw LLM response (handles markdown code blocks), validates criterion names against rubric, clamps scores to scale range
- **`compute_weighted_score()`**: Calculates weighted average from criterion scores using rubric weights
- **`JudgeBackend`** (ABC): Pluggable interface with `evaluate(system_prompt, user_prompt) -> str`
  - `MockJudgeBackend`: Deterministic scores for testing; extracts criterion names via regex
  - `HttpJudgeBackend`: OpenAI-compatible API client with `httpx.AsyncClient`

### Storage (`storage.py`)

- SQLite-backed with indexed queries on `sample_id` and `rubric_name`
- Criterion scores serialized as JSON text column
- Supports in-memory (`:memory:`) for testing and file-based for persistence
- Schema versioned for future migrations

### Core Engine (`core.py`)

- **`EvalEngine.evaluate_sample()`**: Full pipeline -- build prompt, call judge, parse response, compute score, persist
- **`EvalEngine.evaluate_batch()`**: Sequential evaluation with progress logging
- **`EvalEngine.generate_report()`**: Aggregates stored results into `EvalReport` with per-criterion means and score distribution

## Data Flow

```
EvalSample + Rubric
    |
    v
build_judge_prompt()     -- constructs structured evaluation prompt
    |
    v
JudgeBackend.evaluate()  -- sends to LLM, gets raw JSON response
    |
    v
parse_judge_response()   -- extracts JSON, validates, clamps scores
    |
    v
compute_weighted_score() -- weighted average across criteria
    |
    v
EvalResult               -- persisted to SQLite via EvalStorage
    |
    v
generate_report()        -- aggregates into EvalReport with statistics
```

## Extension Points

1. **Custom rubrics**: Define `Rubric` with domain-specific `Criterion` objects, register with `register_rubric()`
2. **Custom judge backends**: Implement `JudgeBackend` ABC for Anthropic, local models, or multi-judge ensembles
3. **Storage backends**: Replace `EvalStorage` with PostgreSQL, DuckDB, or cloud storage
4. **Batch strategies**: Override `evaluate_batch()` for concurrent evaluation with `asyncio.gather()`

## Concurrency Model

The current implementation evaluates samples sequentially within `evaluate_batch()`. The `JudgeBackend.evaluate()` method is async, so upgrading to concurrent evaluation requires only changing the batch loop to use `asyncio.gather()` with a semaphore for rate limiting.
