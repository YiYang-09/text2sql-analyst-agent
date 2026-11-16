"""Provider-agnostic LLM layer with native tool calling.

Two client families cover most providers:
- OpenAICompatClient: any OpenAI-compatible Chat Completions API (DeepSeek, Ollama, Gemini, Groq,
  OpenRouter, OpenAI, vLLM, ...).
- AnthropicClient: Claude via the Anthropic Messages API.

Each client opens a ChatSession that keeps the conversation in the provider's own message format,
so provider-specific content (for example Claude's thinking blocks) is passed back unchanged.
The agent only sees the neutral Turn / ToolCall types below.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    parse_error: str | None = None  # set when the model produced invalid JSON arguments


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    stop_reason: str | None = None


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


class ChatSession(Protocol):
    def send_user(self, text: str) -> Turn: ...
    def send_tool_results(self, results: list[ToolResult], extra_user_text: str | None = None) -> Turn: ...


class LLMClient(Protocol):
    name: str
    model: str

    def start(self, system: str, tools: list[ToolSpec] | None = None) -> ChatSession: ...


# --------------------------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------------------------
class OpenAICompatClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 name: str = "openai-compatible", temperature: float | None = 0.0, max_tokens: int = 4096,
                 timeout_s: float = 120.0):
        from openai import OpenAI  # imported lazily so the package is optional

        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_s, max_retries=2)

    def start(self, system: str, tools: list[ToolSpec] | None = None) -> "OpenAICompatSession":
        return OpenAICompatSession(self, system, tools)


class OpenAICompatSession:
    def __init__(self, client: OpenAICompatClient, system: str, tools: list[ToolSpec] | None):
        self.c = client
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.pending_text_calls = False
        self.n_text_tool_calls = 0
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in (tools or [])
        ]

    def _call(self) -> Turn:
        kwargs: dict[str, Any] = {"model": self.c.model, "messages": self.messages, "max_tokens": self.c.max_tokens}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.c.temperature is not None:
            kwargs["temperature"] = self.c.temperature
        t0 = time.perf_counter()
        resp = self.c.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        msg = resp.choices[0].message
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        calls: list[ToolCall] = []
        self.pending_text_calls = False
        if not msg.tool_calls and self.tools:
            # Some local models (for example qwen2.5-coder in Ollama) write tool calls as JSON text
            # instead of using the tool-call field. Recover them so the loop can continue.
            calls = parse_text_tool_calls(msg.content or "", {t["function"]["name"] for t in self.tools})
            self.pending_text_calls = bool(calls)
            self.n_text_tool_calls += len(calls)
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            for tc in msg.tool_calls:
                args, err = _parse_args(tc.function.arguments)
                calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, parse_error=err))
        self.messages.append(assistant)
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0) if u else Usage()
        return Turn(text=msg.content or "", tool_calls=calls, usage=usage, latency_s=latency,
                    stop_reason=resp.choices[0].finish_reason)

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult], extra_user_text: str | None = None) -> Turn:
        if self.pending_text_calls:
            # The calls came from plain text, so there are no tool-call ids in the history to answer.
            # Return the results as a user message instead.
            parts = [("ERROR: " if r.is_error else "") + r.content for r in results]
            text = "Tool results:\n\n" + "\n\n".join(parts)
            if extra_user_text:
                text += "\n\n" + extra_user_text
            self.messages.append({"role": "user", "content": text})
            return self._call()
        for r in results:
            content = ("ERROR: " + r.content) if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
        if extra_user_text:
            self.messages.append({"role": "user", "content": extra_user_text})
        return self._call()


def parse_text_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Find JSON objects like {"name": "run_sql", "arguments": {...}} in free text."""
    calls: list[ToolCall] = []
    decoder = json.JSONDecoder()
    # Small models sometimes leave the tool name unquoted: {"name": run_sql, ...}. Quote it.
    text = re.sub(r'("name"\s*:\s*)([A-Za-z_]\w*)(\s*[,}])', r'\1"\2"\3', text)
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in tool_names:
            args = obj.get("arguments", obj.get("parameters", {}))
            parsed, err = _parse_args(args)
            calls.append(ToolCall(id=f"text_call_{len(calls) + 1}", name=obj["name"], arguments=parsed, parse_error=err))
            i = end
        else:
            i += 1
    return calls


def _parse_args(raw: str | dict | None) -> tuple[dict, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if not raw:
        return {}, None
    try:
        val = json.loads(raw)
        return (val, None) if isinstance(val, dict) else ({}, "Arguments must be a JSON object.")
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON arguments: {e}"


# --------------------------------------------------------------------------------------------
# Anthropic (Claude)
# --------------------------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 16000, effort: str | None = None,
                 timeout_s: float = 300.0):
        import anthropic  # imported lazily so the package is optional

        self.name = "anthropic"
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s) if api_key else anthropic.Anthropic(timeout=timeout_s)

    def start(self, system: str, tools: list[ToolSpec] | None = None) -> "AnthropicSession":
        return AnthropicSession(self, system, tools)


class AnthropicSession:
    def __init__(self, client: AnthropicClient, system: str, tools: list[ToolSpec] | None):
        self.c = client
        self.system = system
        self.messages: list[dict] = []
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in (tools or [])]

    def _call(self) -> Turn:
        kwargs: dict[str, Any] = {"model": self.c.model, "max_tokens": self.c.max_tokens,
                                  "system": self.system, "messages": self.messages}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.c.effort:
            kwargs["output_config"] = {"effort": self.c.effort}
        t0 = time.perf_counter()
        resp = self.c.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        # Keep the full content (including any thinking blocks) for the next request.
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in resp.content if b.type == "tool_use"]
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return Turn(text=text, tool_calls=calls, usage=usage, latency_s=latency, stop_reason=resp.stop_reason)

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult], extra_user_text: str | None = None) -> Turn:
        content: list[dict] = [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]
        if extra_user_text:
            content.append({"type": "text", "text": extra_user_text})
        self.messages.append({"role": "user", "content": content})
        return self._call()


# --------------------------------------------------------------------------------------------
# Scripted client: replays a fixed policy. Used by tests, CI and the eval-pipeline sanity check.
# --------------------------------------------------------------------------------------------
Policy = Callable[[list[dict]], Turn]


class ScriptedClient:
    """A fake LLM. `policy` receives the transcript so far (list of events) and returns the next Turn."""

    def __init__(self, policy: Policy, name: str = "scripted", model: str = "scripted"):
        self.name = name
        self.model = model
        self.policy = policy

    def start(self, system: str, tools: list[ToolSpec] | None = None) -> "ScriptedSession":
        return ScriptedSession(self.policy, system)


class ScriptedSession:
    def __init__(self, policy: Policy, system: str):
        self.policy = policy
        self.events: list[dict] = [{"type": "system", "text": system}]

    def send_user(self, text: str) -> Turn:
        self.events.append({"type": "user", "text": text})
        return self._next()

    def send_tool_results(self, results: list[ToolResult], extra_user_text: str | None = None) -> Turn:
        for r in results:
            self.events.append({"type": "tool_result", "call_id": r.call_id, "content": r.content, "is_error": r.is_error})
        if extra_user_text:
            self.events.append({"type": "user", "text": extra_user_text})
        return self._next()

    def _next(self) -> Turn:
        turn = self.policy(self.events)
        self.events.append({"type": "assistant", "text": turn.text,
                            "tool_calls": [(c.id, c.name, c.arguments) for c in turn.tool_calls]})
        return turn


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
# Presets for OpenAI-compatible providers. Model names change often: override with LLM_MODEL.
PRESETS: dict[str, dict[str, str | None]] = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:7b", "key_env": None},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash",
               "key_env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": None, "key_env": "OPENROUTER_API_KEY"},
    "openai": {"base_url": None, "model": None, "key_env": "OPENAI_API_KEY"},
}
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"


def make_client(provider: str | None = None, model: str | None = None, base_url: str | None = None,
                api_key: str | None = None) -> LLMClient:
    """Build a client from arguments or environment variables LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    model = model or os.getenv("LLM_MODEL")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    api_key = api_key or os.getenv("LLM_API_KEY")

    if provider == "anthropic":
        return AnthropicClient(model=model or ANTHROPIC_DEFAULT_MODEL,
                               api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                               effort=os.getenv("LLM_EFFORT"))
    if provider in PRESETS:
        p = PRESETS[provider]
        model = model or p["model"]
        if not model:
            raise ValueError(f"Set LLM_MODEL for provider '{provider}'.")
        key = api_key or (os.getenv(p["key_env"]) if p["key_env"] else "ollama")
        if p["key_env"] and not key:
            raise ValueError(f"Set {p['key_env']} (or LLM_API_KEY) for provider '{provider}'.")
        return OpenAICompatClient(model=model, base_url=base_url or p["base_url"], api_key=key, name=provider)
    if provider == "openai-compatible":
        if not (model and base_url):
            raise ValueError("Set LLM_MODEL and LLM_BASE_URL for provider 'openai-compatible'.")
        return OpenAICompatClient(model=model, base_url=base_url, api_key=api_key, name=provider)
    raise ValueError(f"Unknown provider '{provider}'. Use one of: anthropic, {', '.join(PRESETS)}, openai-compatible.")
