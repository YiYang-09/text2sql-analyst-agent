"""Run the benchmark.

Examples:
  python -m evaluation.run_eval --provider oracle                      # pipeline check, no API
  python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode agent
  python -m evaluation.run_eval --provider ollama --mode single_shot --schema ddl --fewshot 0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import time
from collections import defaultdict

from agent.agent import Agent, AgentConfig, SingleShot
from agent.db import Database
from agent.llm import make_client
from evaluation.bench import attach_gold, load_questions
from evaluation.compare import results_match, set_country_names
from evaluation.oracle import agent_oracle, single_shot_oracle

ROOT = pathlib.Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)

    def rate(key):
        return round(sum(r[key] for r in rows) / n, 4) if n else 0.0

    by_diff, by_skill = defaultdict(list), defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r["correct"])
        for s in r["skills"]:
            by_skill[s].append(r["correct"])
    grounded = [r["grounded"] for r in rows if r["grounded"] is not None]
    return {
        "n": n, "correct": k, "execution_accuracy": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "valid_sql_rate": rate("valid_sql"),
        "answered_rate": rate("answered"),
        "avg_sql_errors": rate("n_sql_errors"),
        "avg_llm_calls": rate("n_llm_calls"),
        "grounded_rate": round(sum(grounded) / len(grounded), 4) if grounded else None,
        "avg_latency_s": rate("latency_s"),
        "avg_input_tokens": rate("input_tokens"), "avg_output_tokens": rate("output_tokens"),
        "by_difficulty": {d: round(sum(v) / len(v), 4) for d, v in sorted(by_diff.items())},
        "by_skill": {s: [round(sum(v) / len(v), 4), len(v)] for s, v in sorted(by_skill.items())},
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="oracle", help="oracle, oracle-mistake, ollama, deepseek, gemini, anthropic, ...")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--mode", choices=["agent", "single_shot"], default="agent")
    ap.add_argument("--schema", choices=["semantic", "ddl"], default="semantic")
    ap.add_argument("--fewshot", type=int, default=0, help="number of dev examples in the prompt (0 to 10)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None, help="comma-separated question ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None, help="run name (folder under runs/)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (rate limits)")
    args = ap.parse_args(argv)

    db = Database()
    set_country_names(db.run("select geo, country_name from dim_country").rows)
    questions = attach_gold(load_questions(split=args.split), db)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]
    dev = load_questions(split="dev")
    fewshot = [(q.question, q.sql) for q in dev[: args.fewshot]]
    config = AgentConfig(schema_mode=args.schema, fewshot=fewshot)

    gold_map = {q.question: q.sql for q in questions}
    if args.provider in ("oracle", "oracle-mistake"):
        llm = (single_shot_oracle(gold_map) if args.mode == "single_shot"
               else agent_oracle(gold_map, make_mistake=args.provider == "oracle-mistake"))
    else:
        llm = make_client(args.provider, args.model, args.base_url)

    name = args.name or f"{dt.datetime.now():%Y%m%d-%H%M%S}_{args.provider}_{llm.model.replace('/', '-').replace(':', '-')}" \
                        f"_{args.mode}_{args.schema}_fs{args.fewshot}"
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config.chart_dir = out_dir / "charts"
    runner = Agent(llm, db, config) if args.mode == "agent" else SingleShot(llm, db, config)

    rows = []
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            res = runner.run(q.question)
            fq = res.final_query
            correct = bool(fq and fq.ok and results_match(fq.columns, fq.rows, q.gold.columns, q.gold.rows, q.order_matters))
            row = {
                "id": q.id, "difficulty": q.difficulty, "skills": q.skills, "correct": correct,
                "valid_sql": bool(fq and fq.ok), "answered": res.status == "answered",
                "n_sql_errors": res.n_sql_errors, "n_llm_calls": res.n_llm_calls,
                "grounded": (res.grounding.ok if res.grounding else None),
                "latency_s": round(res.latency_s, 3),
                "input_tokens": res.usage.input_tokens, "output_tokens": res.usage.output_tokens,
                "gold": q.gold.to_dict(), "run": res.to_dict(),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"[{i}/{len(questions)}] {q.id} {'OK ' if correct else 'BAD'} {res.status} "
                  f"sql_err={res.n_sql_errors} {res.latency_s:.1f}s" + (f" {res.error}" if res.error else ""))
            if args.sleep:
                time.sleep(args.sleep)

    summary = summarize(rows)
    summary["config"] = {"provider": args.provider, "model": llm.model, "mode": args.mode, "schema": args.schema,
                         "fewshot": args.fewshot, "split": args.split, "n_questions": len(questions),
                         "date": dt.date.today().isoformat()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_skill"}, indent=2))
    print(f"Saved to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
