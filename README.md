# Text-to-SQL analyst agent

An LLM agent that answers business questions about a real data warehouse by writing and running SQL,
retrying on errors, drawing charts, and explaining the result. It is evaluated on a hand-written
benchmark with gold SQL, so every claim about it comes with a number.

The data is the DuckDB star schema of Eurostat lorry registrations built in my earlier project
**[EU-Truck-Electrification](https://github.com/YiYang-09/EU-Truck-Electrification)** (data model,
41 validation checks and a Power BI dashboard). This repository adds the natural-language layer on top.

> Status: stages 1 and 2 (benchmark and agent) are done and tested with a scripted LLM.
> Model results and the demo are next. Results below will be filled in from `runs/`.

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

### Results

_To be filled in._

| Model | Method | Schema | Few-shot | Execution accuracy | 95 % CI | Avg SQL errors | Grounded |
|---|---|---|---|---|---|---|---|

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_db.py                 # builds data/trucks.duckdb from data/csv
python -m pytest -q                        # 40 tests, no API needed
python -m evaluation.run_eval --provider oracle   # pipeline check, must score 100 %
```

Free, local model with [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:7b
python -m agent.cli "Which EU country had the highest electric share of new lorries in 2024?"
python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode agent
python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode single_shot
```

Any other provider: set `LLM_PROVIDER` (`ollama`, `deepseek`, `gemini`, `groq`, `openrouter`,
`openai`, `anthropic`, `openai-compatible`), `LLM_MODEL`, and the provider's key. See `.env.example`.

## Repository layout

```
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
