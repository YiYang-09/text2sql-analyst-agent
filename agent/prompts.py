"""System prompts and schema context."""
from __future__ import annotations

import pathlib

from agent.db import Database

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEMANTIC_LAYER = ROOT / "data" / "semantic_layer.md"


def schema_context(db: Database, mode: str) -> str:
    """mode 'ddl': table and column names with types only. mode 'semantic': the hand-written semantic layer."""
    if mode == "ddl":
        return "Tables (DuckDB):\n" + db.ddl()
    if mode == "semantic":
        return SEMANTIC_LAYER.read_text(encoding="utf-8")
    raise ValueError(f"Unknown schema mode {mode!r}")


def fewshot_block(examples: list[tuple[str, str]]) -> str:
    if not examples:
        return ""
    parts = ["Examples of questions and correct SQL for this database:"]
    for q, sql in examples:
        parts.append(f"Question: {q}\nSQL:\n{sql.strip()}")
    return "\n\n".join(parts)


AGENT_SYSTEM = """You are a careful data analyst. You answer questions about a DuckDB database of
European lorry registrations by writing and running SQL.

How to work:
1. If you are unsure about tables or columns, call list_tables or describe_table.
2. Call run_sql with one read-only SELECT query (DuckDB dialect). Read the result. If the query fails
   or the result looks wrong (empty, duplicated, includes aggregates it should not), fix it and run again.
3. Optionally call make_chart on a query result when a chart helps.
4. Finish by calling final_answer with:
   - query_id: the id of the query whose result IS the answer (its rows are compared with the truth,
     so return exactly the rows the question asks for: no extra rows, no aggregate rows);
   - answer: 1 to 4 sentences for a business reader.

Rules for the answer text:
- Every number you write must be copied from a query result. Never compute numbers in your head;
  compute differences, ratios and percentages in SQL first. You may round, and you may show a
  fraction as a percentage.
- If the data cannot answer the question (for example the value is missing), say so plainly and
  point query_id at a query that shows the missing (NULL) value.
- Mention caveats from the data when they matter (partial years, small numbers, quality flags).

{schema}

{fewshot}"""


SINGLE_SHOT_SYSTEM = """You are an expert in DuckDB SQL. Write one read-only SELECT query that answers
the user's question about the database below. Return only the SQL in a ```sql code block.
Return exactly the rows the question asks for.

{schema}

{fewshot}"""
