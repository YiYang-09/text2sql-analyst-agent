"""Provider adapters against a local fake server (no network, no key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.agent import Agent
from agent.llm import OpenAICompatClient, parse_text_tool_calls

NAMES = {"run_sql", "final_answer"}


def test_parse_text_tool_calls():
    text = ('{"name": "run_sql", "arguments": {"sql": "select 1"}}\n'
            'then {"name": "final_answer", "arguments": "{\\"query_id\\": \\"q1\\", \\"answer\\": \\"1\\"}"}'
            ' and {"name": "other", "arguments": {}}')
    calls = parse_text_tool_calls(text, NAMES)
    assert [c.name for c in calls] == ["run_sql", "final_answer"]
    assert calls[0].arguments == {"sql": "select 1"} and calls[1].arguments["query_id"] == "q1"
    assert parse_text_tool_calls("no calls here {not json}", NAMES) == []
    # Unquoted tool name, as written by qwen2.5-coder:3b.
    calls = parse_text_tool_calls('{"name": run_sql, "arguments": {"sql": "select 1"}}', NAMES)
    assert [c.name for c in calls] == ["run_sql"] and calls[0].arguments == {"sql": "select 1"}


class _Fake(BaseHTTPRequestHandler):
    replies: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.requests.append(body)
        msg = _Fake.replies[len(_Fake.requests) - 1]
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture
def fake_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _Fake.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_native_tool_calls(db, fake_server):
    sql = "select new_electric from metrics_country_year where geo='DE' and year=2024"
    _Fake.replies = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "run_sql", "arguments": json.dumps({"sql": sql})}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t2", "type": "function", "function": {"name": "final_answer",
                                                          "arguments": json.dumps({"query_id": "q1", "answer": "2,978"})}}]},
    ]
    res = Agent(OpenAICompatClient("m", base_url=fake_server, api_key="x"), db).run("q")
    assert res.status == "answered" and res.final_query.rows == [(2978,)]
    assert _Fake.requests[1]["messages"][-1]["role"] == "tool"


def test_text_tool_calls_fallback(db, fake_server):
    sql = "select new_electric from metrics_country_year where geo='DE' and year=2024"
    _Fake.replies = [
        {"role": "assistant", "content": json.dumps({"name": "run_sql", "arguments": {"sql": sql}})},
        {"role": "assistant", "content": json.dumps({"name": "final_answer", "arguments": {"query_id": "q1", "answer": "2,978"}})},
    ]
    res = Agent(OpenAICompatClient("m", base_url=fake_server, api_key="x"), db).run("q")
    assert res.status == "answered" and res.final_query.rows == [(2978,)]
    last = _Fake.requests[1]["messages"][-1]
    assert last["role"] == "user" and "2978" in last["content"]


def test_load_dotenv(tmp_path, monkeypatch):
    from agent.llm import load_dotenv
    env = tmp_path / ".env"
    env.write_text('﻿# comment\nGEMINI_API_KEY="abc"\nLLM_PROVIDER=gemini\n', encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    load_dotenv(env)
    import os
    assert os.environ["GEMINI_API_KEY"] == "abc" and os.environ["LLM_PROVIDER"] == "ollama"
