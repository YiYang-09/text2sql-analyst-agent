"""Tools the agent can call. All numbers come from DuckDB; the LLM only chooses what to run."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from agent.db import Database, QueryResult
from agent.grounding import GroundingReport, check_grounding
from agent.llm import ToolCall, ToolResult, ToolSpec

TOOL_SPECS = [
    ToolSpec("list_tables", "List the tables in the database with their row counts.",
             {"type": "object", "properties": {}, "additionalProperties": False}),
    ToolSpec("describe_table", "Show the columns, types and three sample rows of one table.",
             {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"],
              "additionalProperties": False}),
    ToolSpec("run_sql", "Run one read-only DuckDB SELECT query. Returns a query id (q1, q2, ...) and the result.",
             {"type": "object",
              "properties": {"sql": {"type": "string", "description": "A single SELECT statement."},
                             "purpose": {"type": "string", "description": "One short sentence: what this query checks."}},
              "required": ["sql"], "additionalProperties": False}),
    ToolSpec("make_chart", "Draw a chart from an earlier query result and save it as a PNG.",
             {"type": "object",
              "properties": {"query_id": {"type": "string"},
                             "kind": {"type": "string", "enum": ["line", "bar", "scatter"]},
                             "x": {"type": "string", "description": "Column for the x axis."},
                             "y": {"type": "string", "description": "Numeric column for the y axis."},
                             "series": {"type": "string", "description": "Optional column that splits lines or bars."},
                             "title": {"type": "string"}},
              "required": ["query_id", "kind", "x", "y", "title"], "additionalProperties": False}),
    ToolSpec("final_answer", "Give the final answer. query_id must be the query whose result answers the question.",
             {"type": "object",
              "properties": {"query_id": {"type": "string"}, "answer": {"type": "string"}},
              "required": ["query_id", "answer"], "additionalProperties": False}),
]


@dataclass
class FinalAnswer:
    answer: str
    query_id: str
    grounding: GroundingReport


@dataclass
class Toolbox:
    db: Database
    question: str = ""
    chart_dir: pathlib.Path | None = None
    max_sql_failures: int = 3          # consecutive failed run_sql calls before the budget is spent
    grounding_retries: int = 1         # how often an ungrounded answer is sent back for correction
    queries: dict[str, QueryResult] = field(default_factory=dict)
    charts: list[str] = field(default_factory=list)
    final: FinalAnswer | None = None
    n_sql_errors: int = 0
    consecutive_failures: int = 0
    grounding_rejections: int = 0

    @property
    def retry_budget_spent(self) -> bool:
        return self.consecutive_failures >= self.max_sql_failures

    def execute(self, call: ToolCall) -> ToolResult:
        if call.parse_error:
            return ToolResult(call.id, call.parse_error, is_error=True)
        fn = getattr(self, f"_tool_{call.name}", None)
        if fn is None:
            return ToolResult(call.id, f"Unknown tool '{call.name}'.", is_error=True)
        try:
            content, is_error = fn(**call.arguments)
        except TypeError as e:
            return ToolResult(call.id, f"Bad arguments for {call.name}: {e}", is_error=True)
        return ToolResult(call.id, content, is_error)

    # ---- tools ------------------------------------------------------------------------------
    def _tool_list_tables(self) -> tuple[str, bool]:
        lines = []
        for t in self.db.tables():
            n = self.db.run(f"select count(*) from {t}").rows[0][0]
            lines.append(f"{t} ({n} rows)")
        return "\n".join(lines), False

    def _tool_describe_table(self, table: str) -> tuple[str, bool]:
        if table not in self.db.tables():
            return f"No table named '{table}'. Tables: {', '.join(self.db.tables())}", True
        cols = "\n".join(f"- {c} {t}" for c, t in self.db.columns(table))
        sample = self.db.run(f"select * from {table} using sample 3 rows (reservoir, 42)").to_markdown()
        return f"Columns of {table}:\n{cols}\n\nSample rows:\n{sample}", False

    def _tool_run_sql(self, sql: str, purpose: str = "") -> tuple[str, bool]:
        if self.retry_budget_spent:
            return ("Retry budget spent: the last queries all failed. Call final_answer now with the best "
                    "successful query, or explain that you could not answer."), True
        normalized = " ".join(sql.split()).rstrip(";").lower()
        for prev_id, prev in self.queries.items():
            if prev.ok and " ".join(prev.sql.split()).rstrip(";").lower() == normalized:
                return (f"You already ran this exact query as {prev_id}. Use that result, change the query, "
                        "or call final_answer."), True
        qid = f"q{len(self.queries) + 1}"
        res = self.db.run(sql)
        res.meta["purpose"] = purpose
        self.queries[qid] = res
        if not res.ok:
            self.n_sql_errors += 1
            self.consecutive_failures += 1
            left = self.max_sql_failures - self.consecutive_failures
            return f"{qid} failed: {res.error}\n(Fix the query and try again; {left} attempt(s) left.)", True
        self.consecutive_failures = 0
        hint = ""
        if not res.rows:
            hint = ("\nHint: the query returned no rows. Before concluding that the data is missing, check "
                    "the filter values (codes are upper case, e.g. 'DE', 'EU27_2020', 'ELC'; call describe_table "
                    "to see sample values).")
        return f"{qid}:\n{res.to_markdown()}{hint}", False

    def _tool_make_chart(self, query_id: str, kind: str, x: str, y: str, title: str, series: str | None = None) -> tuple[str, bool]:
        res = self.queries.get(query_id)
        if res is None or not res.ok:
            return f"No successful query with id {query_id}.", True
        from agent.charts import draw_chart
        out_dir = self.chart_dir or pathlib.Path("outputs")
        try:
            path = draw_chart(res, kind, x, y, title, series, out_dir / f"chart_{len(self.charts) + 1}.png")
        except (KeyError, ValueError) as e:
            return f"Could not draw chart: {e}", True
        self.charts.append(str(path))
        return f"Saved chart to {path}", False

    def _tool_final_answer(self, query_id: str, answer: str) -> tuple[str, bool]:
        res = self.queries.get(query_id)
        if res is None or not res.ok:
            ok_ids = [k for k, v in self.queries.items() if v.ok]
            return f"query_id must be a successful query. Successful ids: {ok_ids or 'none yet'}.", True
        values = [v for r in self.queries.values() if r.ok for row in r.rows for v in row]
        report = check_grounding(answer, values, self.question)
        if not report.ok and self.grounding_rejections < self.grounding_retries:
            self.grounding_rejections += 1
            bad = ", ".join(m.text for m in report.ungrounded)
            return (f"These numbers do not appear in any query result: {bad}. Compute them with run_sql, "
                    "or remove them, then call final_answer again."), True
        self.final = FinalAnswer(answer=answer, query_id=query_id, grounding=report)
        return "Answer recorded.", False


def args_preview(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False)[:500]
