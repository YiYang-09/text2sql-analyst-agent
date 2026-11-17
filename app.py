"""Streamlit demo.

  streamlit run app.py

Live mode: ask a question; the agent writes and runs SQL with the model chosen in the sidebar.
Replay mode: browse recorded benchmark runs (runs/*/results.jsonl or demo/*.jsonl), no model needed.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile

import pandas as pd
import streamlit as st

from agent.agent import Agent, AgentConfig, SingleShot
from agent.db import DEFAULT_DB, Database
from agent.llm import ANTHROPIC_DEFAULT_MODEL, PRESETS, make_client

ROOT = pathlib.Path(__file__).resolve().parent
EXAMPLES = [
    "Which EU27 member state had the highest electric share of new lorries in 2024?",
    "How many new battery-electric lorries were registered in the EU27 in each year from 2019 to 2024? Draw a line chart.",
    "List the top 5 EU27 member states by new battery-electric lorries in 2024, with their counts.",
    "What share of the EU27's new battery-electric lorries in 2024 were registered in Germany?",
    "What was the EU27 electric share of new lorries in 2025?",
]

st.set_page_config(page_title="Text-to-SQL analyst agent", page_icon="🚚", layout="wide")


@st.cache_resource
def get_db() -> Database:
    if not DEFAULT_DB.exists():
        from scripts.build_db import build
        build()
    return Database()


def result_frame(q: dict) -> pd.DataFrame:
    return pd.DataFrame(q["rows"], columns=q["columns"])


def render_run(run: dict, gold: dict | None = None, correct: bool | None = None) -> None:
    """Render one agent run (the dict produced by RunResult.to_dict())."""
    queries = run.get("queries") or []
    status = run.get("status")
    if run.get("answer"):
        st.subheader("Answer")
        st.write(run["answer"])
    elif status == "llm_error":
        st.error(f"The model call failed: {run.get('error')}")
    elif run.get("mode") != "single_shot":
        st.warning(f"The agent did not give a final answer ({status}).")

    fq = run.get("final_query")
    if fq:
        if fq.get("error"):
            st.error(f"The final query failed: {fq['error']}")
        else:
            st.dataframe(result_frame(fq), hide_index=True)
        with st.expander("SQL behind the answer", expanded=True):
            st.code(fq["sql"], language="sql")
    for path in run.get("charts") or []:
        if os.path.exists(path):
            st.image(path)

    cols = st.columns(5)
    cols[0].metric("LLM calls", run.get("n_llm_calls", 0))
    cols[1].metric("SQL errors", run.get("n_sql_errors", 0))
    cols[2].metric("Latency", f"{run.get('latency_s', 0):.1f} s")
    g = run.get("grounding")
    if g is None:
        cols[3].metric("Numbers grounded", "–")
    else:
        cols[3].metric("Numbers grounded", "yes" if not g["ungrounded"] else "no",
                       help="Every number in the answer text must appear in a query result.")
    if correct is not None:
        cols[4].metric("Matches gold answer", "yes" if correct else "no")
    if g and g["ungrounded"]:
        st.warning("Numbers in the answer that are not in any query result: " + ", ".join(g["ungrounded"]))

    if gold is not None:
        with st.expander("Gold answer"):
            st.code(gold["sql"], language="sql")
            st.dataframe(result_frame(gold), hide_index=True)

    trace = run.get("trace") or []
    if trace:
        st.subheader("Steps")
    step = 0
    for ev in trace:
        if ev["type"] == "llm":
            if ev.get("text") and not ev.get("tool_calls"):
                with st.expander(f"Model text ({ev.get('latency_s', 0):.1f} s)"):
                    st.text(ev["text"][:3000])
            continue
        step += 1
        icon = "❌" if ev["is_error"] else "✅"
        with st.expander(f"{step}. {icon} {ev['name']}", expanded=False):
            m = re.match(r"(q\d+)", ev["result"])
            q = None
            if ev["name"] == "run_sql" and m:
                idx = int(m.group(1)[1:]) - 1
                q = queries[idx] if idx < len(queries) else None
            if q:
                st.code(q["sql"], language="sql")
                if q.get("error"):
                    st.error(q["error"])
                else:
                    st.dataframe(result_frame(q), hide_index=True)
                    if q.get("truncated"):
                        st.caption("Result truncated at the row limit.")
            else:
                st.code(ev["args"], language="json")
                st.text(ev["result"])


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------
st.sidebar.title("Text-to-SQL analyst agent")
mode = st.sidebar.radio("Mode", ["Live", "Replay recorded runs"])

if mode == "Live":
    providers = ["ollama", "deepseek", "gemini", "groq", "openrouter", "openai", "anthropic"]
    default_provider = os.getenv("LLM_PROVIDER", "ollama")
    provider = st.sidebar.selectbox("Provider", providers,
                                    index=providers.index(default_provider) if default_provider in providers else 0)
    default_model = os.getenv("LLM_MODEL") or (
        ANTHROPIC_DEFAULT_MODEL if provider == "anthropic" else PRESETS[provider]["model"] or "")
    model = st.sidebar.text_input("Model", value=default_model)
    api_key = ""
    if provider != "ollama":
        api_key = st.sidebar.text_input("API key", type="password",
                                        help="Used for this session only. Leave empty to use the environment variable.")
    method = st.sidebar.radio("Method", ["Agent (tools, retries)", "Single-shot SQL"])
    schema = st.sidebar.radio("Schema context", ["semantic", "ddl"],
                              help="semantic: tables plus business rules. ddl: table and column names only.")
    st.sidebar.caption("Data: Eurostat lorry registrations 2013 to 2025, from the "
                       "[EU-Truck-Electrification](https://github.com/YiYang-09/EU-Truck-Electrification) project.")

    st.title("Ask the lorry data")
    st.caption("The agent writes SQL, runs it read-only on DuckDB, fixes its own errors, and explains the result. "
               "Check the SQL: small models are often wrong.")
    picked = st.selectbox("Example questions", ["(type your own below)"] + EXAMPLES)
    question = st.text_input("Question (English)", value="" if picked.startswith("(") else picked)

    if st.button("Ask", type="primary") and question.strip():
        db = get_db()
        try:
            llm = make_client(provider, model or None, api_key=api_key or None)
        except Exception as e:
            st.error(str(e))
            st.stop()
        cfg = AgentConfig(schema_mode=schema, chart_dir=pathlib.Path(tempfile.mkdtemp(prefix="charts_")))
        runner = Agent(llm, db, cfg) if method.startswith("Agent") else SingleShot(llm, db, cfg)
        with st.spinner("Working..."):
            res = runner.run(question.strip())
        st.session_state["last_run"] = json.loads(json.dumps(res.to_dict(), default=str))
    if "last_run" in st.session_state:
        st.markdown(f"**Question:** {st.session_state['last_run']['question']}")
        render_run(st.session_state["last_run"])

else:
    files = sorted(ROOT.glob("runs/*/results.jsonl")) + sorted(ROOT.glob("demo/*.jsonl"))
    st.title("Recorded benchmark runs")
    if not files:
        st.info("No recorded runs found. Run the benchmark first, for example:\n\n"
                "`python -m evaluation.run_eval --provider ollama --model qwen2.5-coder:7b --mode agent`")
        st.stop()
    labels = {(f.parent.name if f.name == "results.jsonl" else f.stem): f for f in files}
    name = st.sidebar.selectbox("Run", list(labels))
    rows = [json.loads(line) for line in labels[name].read_text(encoding="utf-8").splitlines() if line.strip()]
    n_ok = sum(r["correct"] for r in rows)
    st.caption(f"{name}: {n_ok} of {len(rows)} correct ({100 * n_ok / len(rows):.1f} %)")
    only = st.sidebar.radio("Show", ["all", "wrong only", "correct only"])
    shown = [r for r in rows if only == "all" or (r["correct"] == (only == "correct only"))]
    if not shown:
        st.info("No questions match this filter.")
        st.stop()
    row = st.selectbox("Question", shown,
                       format_func=lambda r: f"{'✅' if r['correct'] else '❌'} {r['id']} [{r['difficulty']}] {r['run']['question']}")
    st.markdown(f"**Question:** {row['run']['question']}")
    render_run(row["run"], gold=row["gold"], correct=row["correct"])
