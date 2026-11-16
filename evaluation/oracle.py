"""Scripted 'oracle' LLMs that know the gold SQL. They check the evaluation pipeline end to end
without any API: an oracle run must score 100 %."""
from __future__ import annotations

import itertools

from agent.llm import ScriptedClient, ToolCall, Turn

_ids = itertools.count(1)


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=args)


def agent_oracle(gold_by_question: dict[str, str], make_mistake: bool = False) -> ScriptedClient:
    """Agent-mode oracle. With make_mistake it first sends a broken query to exercise the retry path."""

    def policy(events: list[dict]) -> Turn:
        question = next(e["text"] for e in events if e["type"] == "user")
        sql = gold_by_question[question]
        n_assistant = sum(1 for e in events if e["type"] == "assistant")
        results = [e for e in events if e["type"] == "tool_result"]
        if make_mistake and n_assistant == 0:
            return Turn(text="", tool_calls=[_call("run_sql", sql="select no_such_column from dim_year")])
        last_ok = [r for r in results if not r["is_error"] and r["content"].startswith("q")]
        if not last_ok:
            return Turn(text="", tool_calls=[_call("run_sql", sql=sql, purpose="gold")])
        qid = last_ok[-1]["content"].split(":")[0]
        return Turn(text="", tool_calls=[_call("final_answer", query_id=qid, answer="See the query result.")])

    return ScriptedClient(policy, name="oracle", model="oracle-mistake" if make_mistake else "oracle")


def single_shot_oracle(gold_by_question: dict[str, str]) -> ScriptedClient:
    def policy(events: list[dict]) -> Turn:
        question = next(e["text"] for e in events if e["type"] == "user")
        return Turn(text=f"```sql\n{gold_by_question[question]}\n```")

    return ScriptedClient(policy, name="oracle", model="oracle")
