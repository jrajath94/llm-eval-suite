# Interview Prep: llm-eval-suite

## Elevator Pitch (30 seconds)

llm-eval-suite is a production evaluation framework that lets teams define custom rubrics with weighted criteria, use any LLM as a judge, and get structured per-criterion scores with reasoning. Unlike academic benchmark suites that test general capability, this targets the quality dimensions production teams actually care about -- helpfulness, safety, code correctness -- with auditable, repeatable assessments.

## Why I Built This

### The Real Motivation

Every conversation I have with teams shipping LLM products circles back to the same pain point: evaluation. They're using HELM or lm-evaluation-harness to run MMLU and HumanEval, but those benchmarks don't tell them if their chatbot responses are actually helpful to users. They need custom rubrics that match their product's quality bar, per-criterion feedback to debug failures, and historical tracking to measure improvement over time. I built this because the gap between academic eval and production eval is where teams waste the most time building bespoke solutions.

### Company-Specific Framing

| Company         | Why This Matters to Them                                                                                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Anthropic       | Constitutional AI requires structured rubrics for evaluating alignment properties. This framework's weighted criteria map directly to safety/helpfulness evaluation of Claude outputs. Judge-based eval is central to RLHF reward modeling. |
| OpenAI          | Shipping GPT to millions requires scalable eval infrastructure. Custom rubrics let product teams define quality bars per use case. The registry pattern mirrors how eval is organized across product lines at scale.                        |
| DeepMind        | Research-grade evaluation needs reproducibility. The structured rubric + score + reasoning triples provide the kind of detailed signal needed for ablation studies and model comparison.                                                    |
| NVIDIA          | GPU-accelerated inference needs quality metrics beyond throughput. This framework can evaluate output quality across different quantization levels and batch sizes to establish quality-performance Pareto curves.                          |
| Google          | Large-scale model evaluation across Gemini variants needs a framework that separates eval dimensions. Weighted criteria enable different quality bars for different product surfaces.                                                       |
| Meta FAIR       | Open-source eval tooling aligns with Meta's open research mission. The pluggable judge backend lets researchers compare different judge models (Llama vs GPT-4) for evaluation consistency.                                                 |
| Citadel/JS/2Sig | LLM-based financial analysis needs domain-specific evaluation (accuracy of numeric claims, regulatory compliance). Custom rubrics with weighted criteria map directly to risk-weighted quality assessment.                                  |

## Architecture Deep-Dive

The system follows a pipeline architecture:

1. **Input**: `EvalSample` (prompt + response + optional reference) + `Rubric` (weighted criteria)
2. **Prompt Construction**: `build_judge_prompt()` in `judge.py` assembles a structured prompt with rubric criteria, scoring guides, the sample, reference answer, and JSON output format instructions
3. **Judge Dispatch**: `JudgeBackend.evaluate()` sends the prompt to an LLM (mock, OpenAI-compatible API, or custom)
4. **Response Parsing**: `parse_judge_response()` extracts JSON (handling markdown code blocks), validates criterion names against the rubric, and clamps scores to the scale range
5. **Scoring**: `compute_weighted_score()` calculates the weighted average using criterion weights from the rubric
6. **Persistence**: `EvalStorage.save_result()` writes to SQLite with indexed queries on `sample_id` and `rubric_name`
7. **Reporting**: `generate_report()` aggregates results into mean scores, per-criterion averages, and score distributions

### Key Design Decisions

| Decision                     | Why                                                                                  | Alternative                     | Tradeoff                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------- |
| Weighted criteria            | Different dimensions matter differently in production (accuracy 3x > conciseness 1x) | Uniform weights                 | Adds configuration complexity, but captures real quality priorities               |
| Score clamping vs retry      | Deterministic, zero-cost handling of judge hallucination                             | Reject and retry with backoff   | May mask systematic judge calibration issues, but saves 2x+ API budget            |
| SQLite (not Postgres)        | Zero-config, embedded, fast for single-node eval workloads                           | PostgreSQL                      | Doesn't scale to multi-node, but eval is typically single-machine                 |
| JSON in text column          | Flexible schema for criterion scores without joins                                   | Normalized relational schema    | Harder to query individual criteria, but simpler writes and schema evolution      |
| Registry pattern for rubrics | Discoverable, import-time validated, shareable across team                           | YAML/JSON config files          | Requires Python code for custom rubrics, but catches errors at import not runtime |
| Sequential batch evaluation  | Simple, predictable, debuggable                                                      | asyncio.gather() with semaphore | Slower for large batches, but avoids rate limiting complexity                     |

### Scaling Analysis

- **Current capacity**: ~8,500 samples/sec end-to-end with mock judge; real throughput bottlenecked by judge LLM API (typically 5-50 req/sec)
- **10x strategy**: `asyncio.gather()` with semaphore in `evaluate_batch()` for concurrent judge calls; connection pool for SQLite writes
- **100x strategy**: Distributed evaluation with Ray actors, each running an `EvalEngine` instance; shared PostgreSQL or DuckDB for storage; result aggregation via MapReduce
- **Bottlenecks**: Judge LLM API latency and rate limits dominate; SQLite single-writer lock for high-write scenarios
- **Cost estimate**: At $0.01/1K tokens (GPT-4 Turbo), evaluating 10K samples with ~500 tokens/prompt = ~$50

## 10 Deep-Dive Interview Questions

### Q1: Walk me through how a sample gets evaluated end-to-end.

**A:** `EvalEngine.evaluate_sample()` in `core.py:54` receives an `EvalSample` and `Rubric`. It calls `build_judge_prompt()` (`judge.py:31`) which assembles a structured prompt including: rubric name/description, criteria with scoring guides, the sample's prompt and response, optional reference answer, and JSON output format instructions. This prompt goes to `JudgeBackend.evaluate()` which returns raw text. `parse_judge_response()` (`judge.py:72`) uses regex to extract JSON (handling markdown code blocks), validates that criterion names match the rubric, and clamps scores to `[scale_min, scale_max]`. `compute_weighted_score()` (`judge.py:129`) calculates the weighted average using each criterion's weight from the rubric. The resulting `EvalResult` is persisted via `EvalStorage.save_result()` which serializes criterion scores as JSON and writes to SQLite.

### Q2: Why LLM-as-judge over traditional metrics like BLEU or ROUGE?

**A:** BLEU and ROUGE measure surface-level similarity to reference answers. They penalize valid paraphrases and can't evaluate open-ended qualities like helpfulness, safety, or reasoning. LLM-as-judge provides per-criterion assessment with explanations, captures semantic quality, and aligns better with human judgments. The tradeoff is cost (API calls) and potential judge bias, but the structured rubric approach constrains the judge to specific evaluation dimensions, reducing variance.

### Q3: What was the hardest bug you hit?

**A:** The JSON extraction from judge responses. Judge LLMs don't always return clean JSON -- sometimes they wrap it in markdown code blocks (`json ... `), add preamble text, or include explanatory text after the JSON. The regex `r'\{[\s\S]*\}'` in `parse_judge_response()` handles this by finding the first complete JSON object in the response. I initially used `json.loads(raw_response)` directly, which failed ~30% of the time with real judge outputs. The fix was a single regex line, but identifying the pattern required analyzing hundreds of judge responses.

### Q4: How would you scale this to 100x?

**A:** Three changes: (1) Replace sequential batch evaluation with `asyncio.gather()` + `asyncio.Semaphore` for concurrent judge calls, respecting API rate limits. (2) Replace SQLite with PostgreSQL or DuckDB for concurrent writes from multiple evaluation workers. (3) Distribute evaluation across Ray actors, each running an `EvalEngine` instance with its own judge client, reporting results to a shared storage backend. The rubric and prompt-building logic is stateless and parallelizes trivially. The bottleneck shifts to judge API throughput, which you solve with multiple API keys or self-hosted judge models.

### Q5: What would you do differently with more time?

**A:** Three things: (1) Multi-judge ensembles -- run the same sample through 3 different judges and aggregate scores to reduce individual judge bias. (2) Async batch evaluation with rate limiting using `asyncio.Semaphore`. (3) A web dashboard (Streamlit or Gradio) for visualizing quality trends over time, comparing rubric versions, and drilling into individual criterion failures.

### Q6: How does this compare to DeepEval or RAGAS?

**A:** DeepEval provides pre-built metrics (faithfulness, hallucination) optimized for RAG pipelines. RAGAS focuses specifically on retrieval-augmented generation evaluation. llm-eval-suite is more general: it provides the rubric framework for defining arbitrary evaluation dimensions. DeepEval and RAGAS are better if you're evaluating RAG specifically; llm-eval-suite is better if you need custom quality dimensions for any LLM application (chatbots, code generation, summarization, content moderation).

### Q7: What are the security implications?

**A:** Three attack surfaces: (1) **Prompt injection via sample content**: Malicious prompt/response pairs could attempt to manipulate the judge LLM. Mitigated by the structured prompt format that clearly delineates the evaluation context. (2) **SQL injection via sample metadata**: All SQLite queries use parameterized statements (`?` placeholders), so metadata values can't inject SQL. (3) **API key exposure**: `HttpJudgeBackend` stores the API key in memory. In production, you'd use environment variables and rotate keys. The judge API call uses HTTPS for transport security.

### Q8: Explain your testing strategy.

**A:** Four layers: (1) **Unit tests** (`test_models.py`, 15 tests) validate data model construction, validation, and edge cases including parametrized tests across importance levels. (2) **Component tests** (`test_rubrics.py`, 11 tests; `test_judge.py`, 12 tests; `test_storage.py`, 6 tests) test each module in isolation with mock dependencies. (3) **Integration tests** (`test_core.py`, 6 tests) test the full pipeline with `MockJudgeBackend` and in-memory SQLite. (4) **Benchmarks** (`bench_core.py`) measure throughput for prompt building, response parsing, SQLite writes, and end-to-end pipeline. Total: 54 tests, 74% line coverage with branch coverage.

### Q9: What are the failure modes?

**A:** (1) **Judge returns invalid JSON**: `parse_judge_response()` raises `JudgeError`, caught by `evaluate_sample()` and re-raised as `EvalSuiteError`. (2) **Judge returns unknown criteria**: Logged as warning, skipped -- partial results still useful. (3) **Judge scores outside range**: Clamped to `[scale_min, scale_max]`, not rejected. (4) **SQLite write failure**: `StorageError` raised, evaluation result lost but engine continues. (5) **API timeout**: `httpx.AsyncClient` has configurable timeout (default 60s), raises `JudgeError` on failure. Detection: structured logging at each pipeline stage; all exceptions include context for debugging.

### Q10: Explain weighted scoring from first principles.

**A:** Each criterion has a weight reflecting its importance (e.g., accuracy=3.0, conciseness=1.0). The weighted score formula is: `overall = sum(score_i * weight_i) / sum(weight_i)`. This is equivalent to normalizing weights to sum to 1.0 and computing a dot product. The denominator uses actual total weight (not rubric total_weight) because some criteria may be missing from the judge response. This prevents division by zero and handles partial scoring gracefully. The result is always in the range `[scale_min, scale_max]` because individual scores are clamped to that range before weighting.

## Complexity Analysis

- **Time**: O(C) per sample evaluation where C = number of criteria (prompt building and parsing are linear in criteria count). Dominated by judge LLM latency (~1-30s per call), not framework computation (~0.1ms per sample).
- **Space**: O(N \* C) for N stored results with C criterion scores each. SQLite database grows linearly.
- **Network**: One HTTP round-trip per sample to judge LLM API. Batch evaluation is N sequential calls (could be concurrent).
- **Disk**: ~500 bytes per result in SQLite (sample_id + rubric + JSON scores + metadata).

## Metrics & Results

| Metric            | Value     | How Measured                          | Significance                                 |
| ----------------- | --------- | ------------------------------------- | -------------------------------------------- |
| Prompt Throughput | 192,384/s | 10K iterations, `time.perf_counter()` | Framework overhead negligible vs API latency |
| Parse Throughput  | 36,752/s  | 10K iterations, `time.perf_counter()` | JSON extraction + validation is fast         |
| SQLite Write      | 26,110/s  | 5K inserts, in-memory DB              | Not the bottleneck even with persistence     |
| E2E Throughput    | 8,483/s   | 100 samples, 3 runs averaged          | Full pipeline overhead ~0.12ms/sample        |
| Test Count        | 54        | pytest                                | Comprehensive across all modules             |
| Coverage          | 74%       | pytest-cov (branch)                   | CLI and utils formatting uncovered           |
| Test Speed        | 0.41s     | Full suite                            | Fast feedback loop                           |

## Career Narrative

How this project fits the story:

- **JPMorgan (current)**: Built ML model validation frameworks with structured rubrics for regulatory compliance -- same pattern of multi-dimensional quality assessment
- **Goldman Sachs (quant)**: Evaluated trading model performance across risk metrics -- weighted scoring across dimensions is directly analogous
- **NVIDIA**: Understood inference optimization tradeoffs -- this framework can evaluate quality-performance Pareto curves across quantization levels
- **This project**: Demonstrates evaluation infrastructure expertise -- the exact capability AI labs need as they scale from research to production deployment

## Interview Red Flags to Avoid

- NEVER say "I built this to learn X" (sounds junior)
- NEVER be unable to explain any line of your code
- NEVER claim metrics you can't reproduce live
- NEVER badmouth existing tools (compare fairly)
- ALWAYS connect to the company's specific challenges
- ALWAYS mention what you'd improve
- ALWAYS discuss failure modes unprompted
