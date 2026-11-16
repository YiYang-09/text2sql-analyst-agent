"""Build the results table and a per-skill breakdown from runs/*/summary.json.

  python -m evaluation.report                 # prints Markdown
  python -m evaluation.report --out results.md
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_runs(runs_dir: pathlib.Path) -> list[dict]:
    runs = []
    for p in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s["config"]["provider"].startswith("oracle") or s["config"]["n_questions"] < 20:
            continue
        s["name"] = p.parent.name
        runs.append(s)
    order = {"single_shot": 0, "agent": 1}
    runs.sort(key=lambda s: (s["config"]["model"], order[s["config"]["mode"]], s["config"]["schema"], s["config"]["fewshot"]))
    return runs


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def results_table(runs: list[dict]) -> str:
    lines = ["| Model | Method | Schema | Few-shot | Execution accuracy | 95 % CI | Easy | Medium | Hard "
             "| Valid SQL | Avg SQL errors | Avg LLM calls | Grounded | Avg latency |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in runs:
        c, d = s["config"], s["by_difficulty"]
        lo, hi = s["ci95"]
        lines.append(
            f"| {c['model']} | {c['mode'].replace('_', '-')} | {c['schema']} | {c['fewshot']} "
            f"| **{pct(s['execution_accuracy'])}** | {100 * lo:.0f} to {100 * hi:.0f}% "
            f"| {pct(d.get('easy'))} | {pct(d.get('medium'))} | {pct(d.get('hard'))} "
            f"| {pct(s['valid_sql_rate'])} | {s['avg_sql_errors']:.2f} | {s['avg_llm_calls']:.1f} "
            f"| {pct(s['grounded_rate'])} | {s['avg_latency_s']:.1f}s |")
    return "\n".join(lines)


def skill_table(runs: list[dict]) -> str:
    skills = sorted({k for s in runs for k in s["by_skill"]})
    head = "| Skill (n) | " + " | ".join(r["name"] for r in runs) + " |"
    lines = [head, "|---|" + "---|" * len(runs)]
    for k in skills:
        n = next(s["by_skill"][k][1] for s in runs if k in s["by_skill"])
        cells = [pct(s["by_skill"].get(k, [None])[0]) for s in runs]
        lines.append(f"| {k} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load_runs(pathlib.Path(args.runs))
    md = "## Results\n\n" + results_table(runs) + "\n\n## Accuracy by skill\n\n" + skill_table(runs) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
