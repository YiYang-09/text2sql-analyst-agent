"""Ask one question from the command line.

  python -m agent.cli "Which EU country had the highest electric share of new lorries in 2024?"
  LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=... python -m agent.cli "..."
"""
from __future__ import annotations

import argparse
import json

from agent.agent import Agent, AgentConfig
from agent.db import Database
from agent.llm import make_client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--schema", choices=["semantic", "ddl"], default="semantic")
    ap.add_argument("--json", action="store_true", help="print the full run as JSON")
    args = ap.parse_args()

    agent = Agent(make_client(args.provider, args.model), Database(), AgentConfig(schema_mode=args.schema))
    res = agent.run(args.question)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str, ensure_ascii=False))
        return
    for step in res.trace:
        if step["type"] == "tool":
            print(f"\n--> {step['name']} {step['args']}")
            print(("  [error] " if step["is_error"] else "  ") + step["result"][:800].replace("\n", "\n  "))
    print("\n" + "=" * 70)
    print(res.answer or f"(no answer: {res.status} {res.error or ''})")
    if res.final_query is not None:
        print("\nSQL:\n" + res.final_query.sql)
    if res.grounding and not res.grounding.ok:
        print("\nWarning: numbers not found in any query result:", [m.text for m in res.grounding.ungrounded])
    print(f"\n{res.n_llm_calls} LLM calls, {res.n_sql_errors} SQL errors, {res.latency_s:.1f}s, "
          f"{res.usage.input_tokens} in / {res.usage.output_tokens} out tokens")


if __name__ == "__main__":
    main()
