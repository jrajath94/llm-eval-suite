# X Thread: llm-eval-suite

**Tweet 1:**
192,000 prompt builds/sec. 36,000 response parses/sec. Zero framework overhead.

I built an LLM evaluation framework that doesn't slow you down.

Code: github.com/jrajath94/llm-eval-suite

**Tweet 2:**
The problem: existing eval frameworks focus on academic benchmarks (MMLU, HumanEval).

But production teams need to answer different questions:

- Is this response helpful?
- Is it safe?
- Does the generated code actually work?

You need custom rubrics, not fixed benchmarks.

**Tweet 3:**
How it works:

1. Define rubrics with weighted criteria (accuracy 3x, conciseness 1x)
2. Any LLM judges the response per-criterion
3. Structured JSON output with scores + reasoning
4. SQLite storage for tracking over time
5. Aggregated reports with per-criterion breakdowns

Architecture: Sample -> Rubric -> Judge -> Parse -> Score -> Store -> Report

**Tweet 4:**
The non-obvious design decision: score clamping instead of retry.

When the judge LLM hallucinates a score of 7 on a 1-5 scale, we clamp to 5.

Alternative: reject and retry. But that wastes API budget and adds latency. Clamping is deterministic and cheap.

**Tweet 5:**
Benchmarks (Apple M2, Python 3.9):

- Prompt building: 192,384/s
- Response parsing: 36,752/s
- SQLite writes: 26,110/s
- End-to-end pipeline: 8,483 samples/s

Your bottleneck is always the judge LLM API, never the framework.

**Tweet 6:**
Star it if you're building LLM products and need production eval.

github.com/jrajath94/llm-eval-suite

3 built-in rubrics (helpfulness, safety, code_quality) + define your own.

#AI #MachineLearning #LLM #OpenSource #BuildInPublic
