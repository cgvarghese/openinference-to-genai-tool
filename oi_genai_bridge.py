"""
oi_genai_bridge.py — OpenInference -> GenAI semconv bridge (in-process).

Wraps any OTel SpanExporter and, before export, adds GenAI semantic-convention
attributes built from the OpenInference ones. OpenInference attributes are left
untouched, and existing gen_ai.* attributes are never overwritten.

Same mapping as otel-collector-config.yaml, but runs in-process (no collector).
Demo wiring: lang-to-genai.py.

  Operation   : root span -> invoke_agent, LLM -> chat, TOOL -> execute_tool
  Grouping    : session.id -> gen_ai.conversation.id
                agent.name (or agent_name arg) -> gen_ai.agent.name (root span)
  Enrichment  : llm.model_name -> gen_ai.request.model, llm.provider -> gen_ai.provider.name,
                llm.token_count.prompt/completion -> gen_ai.usage.input/output_tokens,
                tool.name -> gen_ai.tool.name
  Messages    : llm.input/output_messages.* -> gen_ai.input/output.messages (LLM spans)
                input.value / output.value  -> gen_ai.tool.call.arguments / result (TOOL spans)
                turn's newest user msg + final answer -> gen_ai.input/output.messages (root)

Only dependency: opentelemetry-sdk.

Usage (export straight to the Weave agents endpoint):
    from oi_genai_bridge import OpenInferenceToGenAIExporter
    exporter = OpenInferenceToGenAIExporter(
        OTLPSpanExporter(endpoint="https://trace.wandb.ai/agents/otel/v1/traces",
                         headers={"wandb-api-key": WANDB_API_KEY}),
        agent_name="my-agent",
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))  # or BatchSpanProcessor
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Mapping

try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
except ImportError:  # lets the mapping logic be unit-tested without the SDK
    ReadableSpan = None
    SpanExporter = object
    SpanExportResult = None

OI_KIND = "openinference.span.kind"
ROLE_MAP = {"human": "user", "ai": "assistant", "function": "tool"}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _indices(attrs: Mapping[str, Any], pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return sorted({int(m.group(1)) for k in attrs if (m := rx.match(k))})


def oi_messages(attrs: Mapping[str, Any], prefix: str, is_output: bool = False) -> list[dict]:
    """Rebuild GenAI-format messages from flattened OpenInference attributes."""
    messages = []
    for i in _indices(attrs, rf"^{re.escape(prefix)}\.(\d+)\.message\."):
        base = f"{prefix}.{i}.message"
        role = str(attrs.get(f"{base}.role", "user"))
        role = ROLE_MAP.get(role, role)
        content = attrs.get(f"{base}.content")
        parts: list[dict] = []

        if role == "tool":
            parts.append({
                "type": "tool_call_response",
                "id": attrs.get(f"{base}.tool_call_id"),
                "response": content,
            })
        else:
            if content:
                parts.append({"type": "text", "content": content})
            # multi-part content (message.contents.N.message_content.*)
            for k in _indices(attrs, rf"^{re.escape(base)}\.contents\.(\d+)\.message_content\.text$"):
                text = attrs.get(f"{base}.contents.{k}.message_content.text")
                if text:
                    parts.append({"type": "text", "content": text})
            for j in _indices(attrs, rf"^{re.escape(base)}\.tool_calls\.(\d+)\.tool_call\."):
                tc = f"{base}.tool_calls.{j}.tool_call"
                parts.append({
                    "type": "tool_call",
                    "id": attrs.get(f"{tc}.id"),
                    "name": attrs.get(f"{tc}.function.name"),
                    "arguments": _maybe_json(attrs.get(f"{tc}.function.arguments")),
                })

        msg: dict[str, Any] = {"role": role, "parts": parts}
        if is_output:
            has_tool_call = any(p["type"] == "tool_call" for p in parts)
            msg["finish_reason"] = "tool_call" if has_tool_call else "stop"
        messages.append(msg)
    return messages


def _tool_result(value: Any) -> Any:
    """LangChain often serializes tool output as a ToolMessage; pull out its content."""
    parsed = _maybe_json(value)
    if isinstance(parsed, dict):
        if "content" in parsed:
            return parsed["content"]
        kwargs = parsed.get("kwargs")
        if isinstance(kwargs, dict) and "content" in kwargs:
            return kwargs["content"]
    return parsed


def _dump(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class MessageBridge:
    """Stateful mapper. Feed spans in end-time order (children end before their root)."""

    def __init__(self, agent_name: str | None = None, map_semconv: bool = True, max_traces: int = 10_000):
        self._turns: OrderedDict[str, dict] = OrderedDict()
        self._max = max_traces
        self._agent_name = agent_name
        self._map_semconv = map_semconv

    def _semconv(self, is_root: bool, attrs: Mapping[str, Any]) -> dict[str, Any]:
        """The non-message mappings (what otel-collector-config.yaml does)."""
        kind = attrs.get(OI_KIND)
        out: dict[str, Any] = {}

        op = "invoke_agent" if is_root else {"LLM": "chat", "TOOL": "execute_tool"}.get(kind)
        if op:
            out["gen_ai.operation.name"] = op
        if attrs.get("session.id") is not None:
            out["gen_ai.conversation.id"] = attrs["session.id"]
        agent = attrs.get("agent.name") or (self._agent_name if is_root else None)
        if agent:
            out["gen_ai.agent.name"] = agent

        for src, dst in (
            ("llm.model_name", "gen_ai.request.model"),
            ("llm.provider", "gen_ai.provider.name"),
            ("llm.token_count.prompt", "gen_ai.usage.input_tokens"),
            ("llm.token_count.completion", "gen_ai.usage.output_tokens"),
            ("tool.name", "gen_ai.tool.name"),
        ):
            if attrs.get(src) is not None:
                out[dst] = attrs[src]
        return out

    def _turn(self, trace_id: str) -> dict:
        state = self._turns.setdefault(trace_id, {"input": None, "output": None})
        self._turns.move_to_end(trace_id)
        while len(self._turns) > self._max:
            self._turns.popitem(last=False)
        return state

    def convert(self, trace_id: str, is_root: bool, attrs: Mapping[str, Any]) -> dict[str, Any]:
        """Return gen_ai.* attributes to add (never overwrites keys already on the span)."""
        extra = self._convert(trace_id, is_root, attrs)
        return {k: v for k, v in extra.items() if k not in attrs}

    def _convert(self, trace_id: str, is_root: bool, attrs: Mapping[str, Any]) -> dict[str, Any]:
        kind = attrs.get(OI_KIND)
        extra: dict[str, Any] = self._semconv(is_root, attrs) if self._map_semconv else {}

        if kind == "LLM":
            inputs = oi_messages(attrs, "llm.input_messages")
            outputs = oi_messages(attrs, "llm.output_messages", is_output=True)
            if inputs:
                extra["gen_ai.input.messages"] = _dump(inputs)
            if outputs:
                extra["gen_ai.output.messages"] = _dump(outputs)
            state = self._turn(trace_id)
            if state["input"] is None and inputs:
                users = [m for m in inputs if m["role"] == "user"]
                if users:
                    state["input"] = [users[-1]]
            if outputs:
                state["output"] = outputs

        elif kind == "TOOL":
            if "input.value" in attrs:
                extra["gen_ai.tool.call.arguments"] = _dump(_maybe_json(attrs["input.value"]))
            if "output.value" in attrs:
                extra["gen_ai.tool.call.result"] = _dump(_tool_result(attrs["output.value"]))

        if is_root:
            state = self._turns.pop(trace_id, None)
            if state:
                if state["input"]:
                    extra["gen_ai.input.messages"] = _dump(state["input"])
                if state["output"]:
                    extra["gen_ai.output.messages"] = _dump(state["output"])
        return extra


class OpenInferenceToGenAIExporter(SpanExporter):
    def __init__(self, inner, agent_name: str | None = None, map_semconv: bool = True):
        """map_semconv=False -> messages only (use when a collector already does the rest)."""
        self._inner = inner
        self._bridge = MessageBridge(agent_name=agent_name, map_semconv=map_semconv)

    def export(self, spans):
        out = []
        for span in sorted(spans, key=lambda s: s.end_time or 0):
            extra = self._bridge.convert(
                trace_id=format(span.context.trace_id, "032x"),
                is_root=span.parent is None,
                attrs=span.attributes or {},
            )
            out.append(_with_attributes(span, extra) if extra else span)
        return self._inner.export(out)

    def shutdown(self):
        return self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000):
        return self._inner.force_flush(timeout_millis)


def _with_attributes(span, extra: dict) -> "ReadableSpan":
    attrs = dict(span.attributes or {})
    attrs.update(extra)
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attrs,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )