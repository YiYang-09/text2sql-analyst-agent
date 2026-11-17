# Text-to-SQL analyst agent

An LLM agent that answers business questions about a real data warehouse by writing and running SQL,
retrying on errors, drawing charts, and explaining the result. It is evaluated on a hand-written
benchmark with gold SQL, so every claim about it comes with a number.

The data is the DuckDB star schema of Eurostat lorry registrations built in my earlier project
**[EU-Truck-Electrification](https://github.com/YiYang-09/EU-Truck-Electrification)** (data model,
41 validation checks and a Power BI dashboard). This repository adds the natural-language layer on top.

> Status: benchmark, agent, Streamlit demo and two rounds of local-model results (below) are done.
> Next: an API model with native tool calling as a strong reference, and a hosted demo.

## What it does

```
question ─▶ LLM ─▶ tool calls ─────────────────────────────▶ final_answer
             ▲      list_tables / describe_table               │  checked:
             │      run_sql  (read-only, row limit, timeout)   │  - query id exists and succeeded
             │      make_chart                                 │  - every number in the text
             └──── results / errors (max 3 failed queries) ◀───┘    appears in a query result
```

- **Native tool calling, no agent framework.** The loop in `agent/agent.py` is under 100 lines.
- **Provider-agnostic.** One interface, two adapters: any OpenAI-compatible API (Ollama, DeepSeek,
  Gemini, Groq, OpenRouter, OpenAI) and Anthropic Claude. Switch with an environment variable.
- **Works with models that lack native tool calling.** Some local models (for example
  `qwen2.5-coder` in Ollama) write tool calls as JSON text. The OpenAI-compatible adapter recovers
  them from the text and returns tool results as a user message; runs record how often this happened.
- **No blind answers.** A `final_answer` sent in the same turn as a new query is refused: the model
  must read the result first.
- **Safe SQL.** The database is opened read-only with external file access disabled, and a guard
  accepts only a single `SELECT` statement and blocks file-reading table functions. Results are
  capped at 500 rows and queries time out after 15 seconds.
- **Numbers come from code, not from the LLM.** The final answer must point to the query that
  answers the question, and a grounding check extracts every number from the answer text and
  requires it to appear in an executed result (rounding and fraction-to-percent allowed). An answer
  with an invented or mentally computed number is sent back once for correction.
- **Semantic layer.** `data/semantic_layer.md` documents the tables and the business rules that
  make this data easy to get wrong (see below). Whether the agent gets it is an experiment variable.

## The benchmark

`bench/questions.yaml`: 73 hand-written questions with gold SQL.

| Split | Easy | Medium | Hard | Use |
|---|---|---|---|---|
| dev | 3 | 5 | 2 | few-shot example pool, never scored |
| test | 20 | 25 | 18 | scored |

Each question is tagged with the skills it needs. The data has real traps, and the benchmark tests them:

| Trap | Example of the mistake |
|---|---|
| `trap_aggregate` | The fact tables contain an `EU27_2020` aggregate row. "Which country registered the most electric lorries?" returns the EU27 instead of Germany. |
| `fuel_hierarchy` | Fuel codes have total, subtotal and leaf levels. Summing across levels double counts. |
| `trap_null` | Missing values are not zero; 2025 is incomplete. |
| `unanswerable` | The EU27 share for 2025 does not exist. The right answer is "not available", not a partial sum. |

**How answers are scored (execution accuracy).** The agent's chosen query result is compared with
the gold result (`evaluation/compare.py`): column names and order do not matter and extra columns are
allowed; row order matters only when the question asks for a ranking; the row count must match;
a country may be given as code or name; a share may be a fraction or a percentage; integers must
match exactly and other numbers within 0.1 % or the predicted value's rounding. Tests in
`tests/test_bench.py` check that typical mistakes score as wrong and equivalent answers as right.

## Experiments

| Variable | Values |
|---|---|
| Method | single-shot SQL (one call, no retries) vs agent loop |
| Schema context | table and column names only (`ddl`) vs semantic layer |
| Few-shot | 0 vs 3 to 10 dev examples |
| Model | local open models via Ollama, plus API models where available |

Metrics: execution accuracy with a 95 % Wilson interval, valid-SQL rate, average SQL errors and
LLM calls per question, grounding rate, latency and tokens. With 63 test questions the 95 % interval is
about ±12 points around 50 % accuracy and about ±8 points around 85 %, so small differences are reported as such.

### Results, round 1 (local models, 2026-11-16)

Local models via Ollama on an RTX 4060 laptop GPU (8 GB), 63 test questions, no few-shot, code at
commit `661f715`. Accuracy is execution accuracy with a 95 % Wilson interval.

| Model | Method | Schema | Accuracy | 95 % CI | Valid SQL | Avg LLM calls | Grounded | Avg latency |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder 7B | single-shot | ddl | 52.4 % | 40 to 64 % | 90.5 % | 1.0 | – | 2.0 s |
| qwen2.5-coder 7B | single-shot | semantic | **60.3 %** | 48 to 71 % | 82.5 % | 1.0 | – | 2.0 s |
| qwen2.5-coder 7B | agent | ddl | 49.2 % | 37 to 61 % | 98.4 % | 3.8 | 96.4 % | 13.5 s |
| qwen2.5-coder 7B | agent | semantic | 54.0 % | 42 to 66 % | 88.9 % | 4.5 | 96.0 % | 10.8 s |
| qwen2.5-coder 3B | single-shot | ddl | 25.4 % | 16 to 37 % | 68.3 % | 1.0 | – | 1.2 s |
| qwen2.5-coder 3B | single-shot | semantic | 46.0 % | 34 to 58 % | 69.8 % | 1.0 | – | 1.2 s |
| qwen2.5-coder 3B | agent | ddl | 23.8 % | 15 to 36 % | 76.2 % | 7.3 | 100 % | 9.9 s |
| qwen2.5-coder 3B | agent | semantic | 44.4 % | 33 to 57 % | 71.4 % | 5.6 | 95.2 % | 6.2 s |

What the numbers say:

- **The semantic layer is the biggest lever.** It lifts the 3B model by 21 points in single-shot mode
  (25 % to 46 %, intervals barely overlap) and helps the 7B model by 8 points (within noise). Most of
  the gain is on the trap questions: 7B single-shot goes from 29 % to 71 % on `trap_aggregate`.
- **The agent loop does not beat single-shot with these small models.** The differences are inside the
  intervals for both models. The loop does raise the share of valid SQL (7B, ddl: 90 % to 98 %): it
  fixes syntax errors, not logic errors.
- **Grounding works as intended.** 95 % or more of the agent answers contain only numbers that
  appear in a query result.

### Failure analysis, round 1

**Local models without native tool calling.** `qwen2.5-coder` in Ollama writes tool calls as JSON text
instead of the tool-call field, so the first agent run executed no SQL at all (0 of 5 in a smoke test,
including an invented answer). The adapter now parses tool calls from text. The 3B model then wrote
`{"name": run_sql, ...}` with an unquoted name, which is not valid JSON; with a lenient parser its agent
accuracy went from 6 % / 3 % to 24 % / 44 %.

Where the 7B agent lost points that single-shot got right (8 questions with ddl, 9 with semantic):

| Pattern | Example |
|---|---|
| Points `final_answer` at a helper query | t36: answered "not available" correctly but cited `select count(*) from dim_country ...` |
| Gives up on an empty result | t27: filtered `geo = 'eu27'` (lower case), got no rows, answered "the data does not provide this" |
| Picks the wrong column | t05: used `new_electric_share` for a question about lorries *in use* |
| Right content, wrong shape | t10: answered "2013 to 2025" with `min`/`max` instead of listing the years |
| Runs out of turns | 7B semantic: 10 of 63 questions hit the 12-turn limit (3B ddl: 31) |

Agent v2 addresses the first three directly: an empty result now comes back with a hint to check
filter values, a repeated identical query is refused, and the prompt states that `query_id` must be the
answering query and that `final_answer` is required even for "not available". These changes come from
reading test-set failures, so v2 results will be reported next to v1, not instead of it.

### Round 2: agent v2 (2026-11-17)

Same models, questions and settings; agent code at `42c0c60` (run at `89b00ce`).

| Model | Schema | Agent v1 | Agent v2 | 95 % CI (v2) | Hit turn limit v1 → v2 | Single-shot |
|---|---|---|---|---|---|---|
| qwen2.5-coder 7B | ddl | 49.2 % | 52.4 % | 40 to 64 % | 6 → 2 | 52.4 % |
| qwen2.5-coder 7B | semantic | 54.0 % | 50.8 % | 39 to 63 % | 10 → 7 | 60.3 % |
| qwen2.5-coder 3B | ddl | 23.8 % | 28.6 % | 19 to 41 % | 31 → 33 | 25.4 % |
| qwen2.5-coder 3B | semantic | 44.4 % | 44.4 % | 33 to 57 % | 21 → 21 | 46.0 % |

**No measurable improvement.** v2 fixed most of the targeted failures (t05 and t36 now pass in both
schemas, t27 and t10 in one), and the 7B model runs out of turns less often. But in every configuration
v2 also got a similar number of other questions newly wrong (for example 7B ddl: 5 newly right, 3 newly
wrong), and all changes are well inside the intervals. With 7B-class models the bottleneck is the
model, not the loop: question-level results move between runs by about as much as the fixes gain.
The next step is a stronger model with native tool calling, to see whether the loop helps when the
model can use it.

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_db.py                 # builds data/trucks.duckdb from data/csv
python -m pytest -q                        # 41 tests, no API needed
python -m evaluation.run_eval --provider oracle   # pipeline check, must score 100 %
```

Free, local model with [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:7b
python -m agent.cli "Which EU country had the highest electric share of new lorries in 2024?"
python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode agent
python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode single_shot
```

Web UI (live questions, or replay of recorded benchmark runs with the gold answer side by side):

```bash
streamlit run app.py
```

Any other provider: set `LLM_PROVIDER` (`ollama`, `deepseek`, `gemini`, `groq`, `openrouter`,
`openai`, `anthropic`, `openai-compatible`), `LLM_MODEL`, and the provider's key. See `.env.example`.

## Repository layout

```
app.py        Streamlit demo: live mode and replay of recorded runs
agent/        llm.py (provider adapters), agent.py (loop + single-shot baseline), tools.py,
              db.py (read-only DuckDB + SQL guard), grounding.py, prompts.py, charts.py, cli.py
bench/        questions.yaml (benchmark with gold SQL)
data/         csv/ (from EU-Truck-Electrification), semantic_layer.md
evaluation/   run_eval.py, compare.py (execution accuracy), oracle.py (scripted pipeline check)
scripts/      build_db.py
tests/        guard, comparator, grounding, benchmark traps, agent loop with a scripted LLM
```

## Data

Eurostat tables `road_eqr_lormot` (new lorries by motor energy) and `road_eqs_lormot` (lorries in use),
2013 to 2025, lorries over 3.5 t, as modelled in
[EU-Truck-Electrification](https://github.com/YiYang-09/EU-Truck-Electrification). Eurostat data may be
reused with attribution.

## Licence

MIT (see `LICENSE`). The data belongs to Eurostat.
