"""
oi_genai_bridge.py — OpenInference -> GenAI semconv bridge (in-process).

Wraps any OTel SpanExporter and, before export, adds GenAI semantic-convention
attributes built from the OpenInference ones. OpenInference attributes are left
untouched, and existing gen_ai.* attributes are never overwritten.

Same mapping as otel-collector-config.yaml, but runs in-process (no collector).
Demo wiring: lang-to-genai.py (live LangGraph) and mock-trace-to-genai.py (replays
mock_trace.json, a multi-agent supervisor trace).

  Operation   : root span or AGENT -> invoke_agent, LLM -> chat,
                TOOL -> execute_tool, RETRIEVER -> execute_tool (gen_ai.tool.type=datastore)
                CHAIN (non-root) -> left as-is
  Grouping    : session.id -> gen_ai.conversation.id
                agent.name / metadata["agent.name"] / agent_name arg (root) / span name (AGENT)
                  -> gen_ai.agent.name (agent spans)
  Enrichment  : llm.model_name -> gen_ai.request.model, llm.provider (or llm.system) -> gen_ai.provider.name,
                llm.invocation_parameters -> gen_ai.request.temperature / max_tokens / top_p / ...,
                llm.token_count.prompt/completion -> gen_ai.usage.input/output_tokens,
                tool.name / tool.description -> gen_ai.tool.name / gen_ai.tool.description,
                exception event / ERROR status -> error.type
  Messages    : llm.input/output_messages.* -> gen_ai.input/output.messages (LLM spans)
                llm.tools.N.tool.json_schema -> gen_ai.tool.definitions (LLM spans)
                input.value / output.value  -> gen_ai.tool.call.arguments / result (TOOL spans)
                input.value / retrieval.documents.* -> gen_ai.tool.call.arguments / result (RETRIEVER spans)
                input.value / output.value  -> gen_ai.input/output.messages (agent spans),
                  falling back to the turn's newest user msg + final LLM answer on the root

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
    from opentelemetry.trace import StatusCode
except ImportError:  # lets the mapping logic be unit-tested without the SDK
    ReadableSpan = None
    SpanExporter = object
    SpanExportResult = None
    StatusCode = None

OI_KIND = "openinference.span.kind"
ROLE_MAP = {"human": "user", "ai": "assistant", "function": "tool"}
# llm.invocation_parameters key -> gen_ai.request.* attribute
REQUEST_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "seed": "gen_ai.request.seed",
}


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


def oi_tool_definitions(attrs: Mapping[str, Any]) -> list[dict]:
    """llm.tools.N.tool.json_schema -> gen_ai.tool.definitions entries."""
    defs = []
    for i in _indices(attrs, r"^llm\.tools\.(\d+)\.tool\.json_schema$"):
        schema = _maybe_json(attrs[f"llm.tools.{i}.tool.json_schema"])
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function", schema)  # OpenAI-style {"type":"function","function":{...}} or bare
        defs.append({"type": "function", **{k: fn[k] for k in ("name", "description", "parameters") if k in fn}})
    return defs


def oi_documents(attrs: Mapping[str, Any]) -> list[dict]:
    """retrieval.documents.N.document.* -> [{id, content, score, metadata}]."""
    docs = []
    for i in _indices(attrs, r"^retrieval\.documents\.(\d+)\.document\."):
        base = f"retrieval.documents.{i}.document"
        doc = {k: attrs[f"{base}.{k}"] for k in ("id", "content", "score") if f"{base}.{k}" in attrs}
        if f"{base}.metadata" in attrs:
            doc["metadata"] = _maybe_json(attrs[f"{base}.metadata"])
        docs.append(doc)
    return docs


def _value_messages(value: Any, mime: Any, role: str) -> list[dict]:
    """input.value / output.value on an agent span -> one GenAI message.

    Plain text becomes a single message. JSON with a "messages" list (LangGraph state)
    yields the last message with the wanted role (newest user msg / final answer).
    """
    if value is None or value == "":
        return []
    parsed = _maybe_json(value) if mime != "text/plain" else value
    if isinstance(parsed, str):
        text = parsed
    else:
        msgs = parsed.get("messages") if isinstance(parsed, dict) else None
        if not isinstance(msgs, list):
            return []
        text = None
        for m in reversed(msgs):
            if isinstance(m, dict) and isinstance(m.get("kwargs"), dict):  # LangChain dumpd()
                m = m["kwargs"]
            elif isinstance(m, dict) and isinstance(m.get("data"), dict):  # message_to_dict(): {"type", "data"}
                m = {"type": m.get("type"), **m["data"]}
            if isinstance(m, (list, tuple)) and len(m) == 2:               # ("user", "hi")
                m = {"role": m[0], "content": m[1]}
            if not isinstance(m, dict):
                continue
            r = ROLE_MAP.get(m.get("role") or m.get("type"), m.get("role") or m.get("type"))
            if r == role and isinstance(m.get("content"), str) and m["content"]:
                text = m["content"]
                break
        if text is None:
            return []
    msg: dict[str, Any] = {"role": role, "parts": [{"type": "text", "content": text}]}
    if role == "assistant":
        msg["finish_reason"] = "stop"
    return [msg]


def _agent_name(attrs: Mapping[str, Any]) -> Any:
    """agent.name attribute, or agent.name inside the OI `metadata` JSON blob."""
    if attrs.get("agent.name"):
        return attrs["agent.name"]
    meta = _maybe_json(attrs.get("metadata"))
    if isinstance(meta, dict):
        # lc_agent_name: set automatically by langchain create_agent(name=...)
        return meta.get("agent.name") or meta.get("agent_name") or meta.get("lc_agent_name")
    return None


def _tool_result(value: Any) -> Any:
    """LangChain often serializes tool output as a ToolMessage; pull out its content."""
    parsed = _maybe_json(value)
    if isinstance(parsed, dict):
        if "content" in parsed:
            return parsed["content"]
        for key in ("kwargs", "data"):  # dumpd() / message_to_dict() shapes
            inner = parsed.get(key)
            if isinstance(inner, dict) and "content" in inner:
                return inner["content"]
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

    def _semconv(self, is_root: bool, attrs: Mapping[str, Any], name: str | None,
                 error_type: str | None) -> dict[str, Any]:
        """The non-message mappings (what otel-collector-config.yaml does)."""
        kind = attrs.get(OI_KIND)
        is_agent = is_root or kind == "AGENT"
        out: dict[str, Any] = {}

        op = "invoke_agent" if is_agent else {
            "LLM": "chat", "TOOL": "execute_tool", "RETRIEVER": "execute_tool",
        }.get(kind)
        if op:
            out["gen_ai.operation.name"] = op
        if attrs.get("session.id") is not None:
            out["gen_ai.conversation.id"] = attrs["session.id"]
        agent = attrs.get("agent.name") or (is_agent and (
            _agent_name(attrs) or (self._agent_name if is_root else None) or (name if kind == "AGENT" else None)))
        if agent:
            out["gen_ai.agent.name"] = agent

        for src, dst in (
            ("llm.model_name", "gen_ai.request.model"),
            ("llm.provider", "gen_ai.provider.name"),
            ("llm.system", "gen_ai.provider.name"),
            ("llm.token_count.prompt", "gen_ai.usage.input_tokens"),
            ("llm.token_count.completion", "gen_ai.usage.output_tokens"),
            ("tool.name", "gen_ai.tool.name"),
            ("tool.description", "gen_ai.tool.description"),
        ):
            if attrs.get(src) is not None and dst not in out:
                out[dst] = attrs[src]

        if kind == "LLM":
            params = _maybe_json(attrs.get("llm.invocation_parameters"))
            if isinstance(params, dict):
                for src, dst in REQUEST_PARAMS.items():
                    if isinstance(params.get(src), (int, float)) and dst not in out:
                        out[dst] = params[src]
        elif kind == "TOOL":
            out["gen_ai.tool.type"] = "function"
        elif kind == "RETRIEVER":
            out["gen_ai.tool.type"] = "datastore"
            if name and "gen_ai.tool.name" not in out:
                out["gen_ai.tool.name"] = name

        if error_type:
            out["error.type"] = error_type
        return out

    def _turn(self, trace_id: str) -> dict:
        state = self._turns.setdefault(trace_id, {"input": None, "output": None})
        self._turns.move_to_end(trace_id)
        while len(self._turns) > self._max:
            self._turns.popitem(last=False)
        return state

    def convert(self, trace_id: str, is_root: bool, attrs: Mapping[str, Any],
                name: str | None = None, error_type: str | None = None) -> dict[str, Any]:
        """Return gen_ai.* attributes to add (never overwrites keys already on the span)."""
        extra = self._convert(trace_id, is_root, attrs, name, error_type)
        return {k: v for k, v in extra.items() if k not in attrs}

    def _convert(self, trace_id: str, is_root: bool, attrs: Mapping[str, Any],
                 name: str | None, error_type: str | None) -> dict[str, Any]:
        kind = attrs.get(OI_KIND)
        extra: dict[str, Any] = self._semconv(is_root, attrs, name, error_type) if self._map_semconv else {}

        if kind == "LLM":
            inputs = oi_messages(attrs, "llm.input_messages")
            outputs = oi_messages(attrs, "llm.output_messages", is_output=True)
            if inputs:
                extra["gen_ai.input.messages"] = _dump(inputs)
            if outputs:
                extra["gen_ai.output.messages"] = _dump(outputs)
            tools = oi_tool_definitions(attrs)
            if tools:
                extra["gen_ai.tool.definitions"] = _dump(tools)
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

        elif kind == "RETRIEVER":
            # No GenAI retrieval op yet, so treat it as a datastore tool call:
            # query -> arguments, retrieved docs (content + score) -> result.
            if "input.value" in attrs:
                extra["gen_ai.tool.call.arguments"] = _dump(_maybe_json(attrs["input.value"]))
            docs = oi_documents(attrs)
            if docs:
                extra["gen_ai.tool.call.result"] = _dump(docs)
            elif "output.value" in attrs:
                extra["gen_ai.tool.call.result"] = _dump(_maybe_json(attrs["output.value"]))

        if is_root or kind == "AGENT":
            # Agent span's own input/output first (plain-text query / final answer);
            # on the root, fall back to what the turn's LLM spans saw.
            state = self._turns.pop(trace_id, None) if is_root else None
            inputs = _value_messages(attrs.get("input.value"), attrs.get("input.mime_type"), "user")
            outputs = _value_messages(attrs.get("output.value"), attrs.get("output.mime_type"), "assistant")
            inputs = inputs or (state or {}).get("input")
            outputs = outputs or (state or {}).get("output")
            if inputs:
                extra["gen_ai.input.messages"] = _dump(inputs)
            if outputs:
                extra["gen_ai.output.messages"] = _dump(outputs)
        return extra


def _error_type(span) -> str | None:
    """exception.type from the span's exception event, else a prefix of the ERROR status message."""
    for ev in span.events or ():
        if ev.name == "exception" and (ev.attributes or {}).get("exception.type"):
            return str(ev.attributes["exception.type"])
    status = span.status
    if StatusCode is not None and status is not None and status.status_code == StatusCode.ERROR:
        desc = status.description or ""
        return desc.split(":", 1)[0].strip() or "_OTHER"
    return None


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
                name=span.name,
                error_type=_error_type(span),
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
