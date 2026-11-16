"""The agent loop and the single-shot baseline.

Agent loop:  question -> LLM -> tool calls (inspect schema, run SQL, chart) -> results back to the LLM
             -> ... -> final_answer (checked: query exists, numbers grounded).
Baseline:    question -> LLM writes one SQL query -> run it once. No tools, no retries.
"""
from __future__ import annotations

import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agent.db import Database, QueryResult
from agent.grounding import GroundingReport
from agent.llm import LLMClient, ToolResult, Usage
from agent.prompts import AGENT_SYSTEM, SINGLE_SHOT_SYSTEM, fewshot_block, schema_context
from agent.tools import TOOL_SPECS, Toolbox, args_preview


@dataclass
class AgentConfig:
    schema_mode: str = "semantic"       # "semantic" or "ddl"
    fewshot: list[tuple[str, str]] = field(default_factory=list)
    max_turns: int = 12                 # LLM calls per question
    max_sql_failures: int = 3           # consecutive SQL errors allowed
    grounding_retries: int = 1
    chart_dir: pathlib.Path | None = None


@dataclass
class RunResult:
    question: str
    mode: str
    answer: str = ""
    final_query: QueryResult | None = None
    queries: list[QueryResult] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    status: str = "answered"            # answered | no_final_answer | max_turns | llm_error
    n_llm_calls: int = 0
    n_sql_errors: int = 0
    grounding: GroundingReport | None = None
    grounding_rejections: int = 0
    text_tool_calls: int = 0            # tool calls recovered from plain text (models without native tool calling)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "mode": self.mode, "answer": self.answer, "status": self.status,
            "final_query": self.final_query.to_dict() if self.final_query else None,
            "queries": [q.to_dict() for q in self.queries], "charts": self.charts, "trace": self.trace,
            "n_llm_calls": self.n_llm_calls, "n_sql_errors": self.n_sql_errors,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "grounding_rejections": self.grounding_rejections, "text_tool_calls": self.text_tool_calls,
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
            "latency_s": round(self.latency_s, 3), "error": self.error,
        }


class Agent:
    def __init__(self, llm: LLMClient, db: Database, config: AgentConfig | None = None):
        self.llm = llm
        self.db = db
        self.config = config or AgentConfig()

    def system_prompt(self) -> str:
        return AGENT_SYSTEM.format(schema=schema_context(self.db, self.config.schema_mode),
                                   fewshot=fewshot_block(self.config.fewshot)).strip()

    def run(self, question: str) -> RunResult:
        cfg = self.config
        out = RunResult(question=question, mode="agent")
        box = Toolbox(db=self.db, question=question, chart_dir=cfg.chart_dir,
                      max_sql_failures=cfg.max_sql_failures, grounding_retries=cfg.grounding_retries)
        t0 = time.perf_counter()
        session = None
        try:
            session = self.llm.start(self.system_prompt(), TOOL_SPECS)
            turn = session.send_user(question)
            nudged = False
            while True:
                out.n_llm_calls += 1
                out.usage.add(turn.usage)
                out.trace.append({"type": "llm", "text": turn.text, "latency_s": round(turn.latency_s, 3),
                                  "tool_calls": [{"name": c.name, "args": c.arguments} for c in turn.tool_calls]})
                if box.final is not None:
                    break
                if out.n_llm_calls >= cfg.max_turns:
                    out.status = "max_turns"
                    break
                if not turn.tool_calls:
                    if nudged:
                        out.status = "no_final_answer"
                        out.answer = turn.text
                        break
                    nudged = True
                    turn = session.send_user("Use the tools to answer, and finish by calling final_answer.")
                    continue
                results = []
                ran_sql = False
                for call in turn.tool_calls:
                    if box.final is not None:  # ignore calls after an accepted final answer
                        break
                    if call.name == "final_answer" and ran_sql:
                        # Do not accept an answer written before the model has seen the query result.
                        r = ToolResult(call.id, "Read the query result first, then call final_answer "
                                                "in your next message.", is_error=True)
                    else:
                        r = box.execute(call)
                    ran_sql = ran_sql or call.name == "run_sql"
                    results.append(r)
                    out.trace.append({"type": "tool", "name": call.name, "args": args_preview(call.arguments),
                                      "is_error": r.is_error, "result": r.content[:2000]})
                if box.final is not None:
                    break
                extra = None
                if box.retry_budget_spent:
                    extra = "The retry budget is spent. Call final_answer now."
                turn = session.send_tool_results(results, extra)
        except Exception as e:  # provider errors, network, rate limits
            out.status = "llm_error"
            out.error = f"{type(e).__name__}: {e}"
        out.latency_s = time.perf_counter() - t0
        out.queries = list(box.queries.values())
        out.charts = box.charts
        out.n_sql_errors = box.n_sql_errors
        out.grounding_rejections = box.grounding_rejections
        out.text_tool_calls = getattr(session, "n_text_tool_calls", 0)
        if box.final is not None:
            out.status = "answered"
            out.answer = box.final.answer
            out.final_query = box.queries[box.final.query_id]
            out.grounding = box.final.grounding
        elif out.status in ("max_turns", "no_final_answer"):
            # Fall back to the last successful query so the result can still be scored.
            ok = [q for q in out.queries if q.ok]
            out.final_query = ok[-1] if ok else None
        return out


_SQL_BLOCK = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    m = _SQL_BLOCK.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


class SingleShot:
    """Baseline: one LLM call that writes one SQL query, executed once."""

    def __init__(self, llm: LLMClient, db: Database, config: AgentConfig | None = None):
        self.llm = llm
        self.db = db
        self.config = config or AgentConfig()

    def system_prompt(self) -> str:
        return SINGLE_SHOT_SYSTEM.format(schema=schema_context(self.db, self.config.schema_mode),
                                         fewshot=fewshot_block(self.config.fewshot)).strip()

    def run(self, question: str) -> RunResult:
        out = RunResult(question=question, mode="single_shot")
        t0 = time.perf_counter()
        try:
            turn = self.llm.start(self.system_prompt(), None).send_user(question)
            out.n_llm_calls = 1
            out.usage.add(turn.usage)
            out.trace.append({"type": "llm", "text": turn.text, "latency_s": round(turn.latency_s, 3), "tool_calls": []})
            res = self.db.run(extract_sql(turn.text))
            out.queries = [res]
            out.final_query = res
            out.n_sql_errors = 0 if res.ok else 1
        except Exception as e:
            out.status = "llm_error"
            out.error = f"{type(e).__name__}: {e}"
        out.latency_s = time.perf_counter() - t0
        return out
