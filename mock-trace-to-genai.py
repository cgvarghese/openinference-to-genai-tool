"""
mock-trace-to-genai.py — replay mock_trace.json (OpenInference, multi-agent supervisor) through
oi_genai_bridge -> Weave Agents. No LLM calls; the spans are rebuilt from the JSON.

  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
  DRY_RUN=1 python mock-trace-to-genai.py             # print gen_ai.* only, no upload
  export WANDB_API_KEY=... WANDB_ENTITY=... WANDB_PROJECT=...
  python mock-trace-to-genai.py                       # bridged -> Agents tab
  RAW=1 python mock-trace-to-genai.py                 # untouched OI -> Agents tab (stays empty)
  TRACE=other.json python mock-trace-to-genai.py      # replay a different export of the same shape

Each run gets a fresh trace id + session id and is shifted to end "now", so re-runs show up as
new conversations instead of colliding with earlier uploads (KEEP_IDS=1 to disable).
"""

import json
import os
import random
import uuid
from datetime import datetime, timezone

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from oi_genai_bridge import OpenInferenceToGenAIExporter

DRY_RUN = os.getenv("DRY_RUN") == "1"
RAW = os.getenv("RAW") == "1"
KEEP_IDS = os.getenv("KEEP_IDS") == "1"
TRACE = os.getenv("TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_trace.json"))
AGENTS_URL = "https://trace.wandb.ai/agents/otel/v1/traces"


def ns(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)


def load_spans(path: str, resource: Resource) -> list[ReadableSpan]:
    with open(path) as f:
        raw = json.load(f)

    trace_id = random.getrandbits(128) if not KEEP_IDS else None
    shift = 0 if KEEP_IDS else (
        int(datetime.now(timezone.utc).timestamp() * 1e9) - max(ns(s["end_time"]) for s in raw))
    session = {}  # old session.id -> new one
    scope = InstrumentationScope("mock-trace-replay")

    def ctx(span_id_hex: str, tid: int) -> SpanContext:
        return SpanContext(tid, int(span_id_hex, 16), is_remote=False, trace_flags=TraceFlags(TraceFlags.SAMPLED))

    spans = []
    for s in raw:
        tid = trace_id or int(s["trace_id"], 16)
        attrs = dict(s.get("attributes", {}))
        if not KEEP_IDS and "session.id" in attrs:
            attrs["session.id"] = session.setdefault(attrs["session.id"], f"mock-{uuid.uuid4().hex[:8]}")
        code = {"OK": StatusCode.OK, "ERROR": StatusCode.ERROR}.get(s.get("status_code"), StatusCode.UNSET)
        spans.append(ReadableSpan(
            name=s["name"],
            context=ctx(s["span_id"], tid),
            parent=ctx(s["parent_id"], tid) if s.get("parent_id") else None,
            resource=resource,
            attributes=attrs,
            events=[Event(e["name"], e.get("attributes", {}), ns(e["timestamp"]) + shift) for e in s.get("events", [])],
            kind=SpanKind.INTERNAL,
            status=Status(code, s.get("status_message") if code == StatusCode.ERROR else None),
            start_time=ns(s["start_time"]) + shift,
            end_time=ns(s["end_time"]) + shift,
            instrumentation_scope=scope,
        ))
    return spans


class Print(SpanExporter):
    """Prints the tree with each span's gen_ai.* / error.type attrs (all attrs when RAW), then forwards."""

    def __init__(self, inner=None):
        self.inner = inner

    def export(self, spans):
        by_id = {s.context.span_id: s for s in spans}

        def depth(s):
            d = 0
            while s.parent and s.parent.span_id in by_id:
                s, d = by_id[s.parent.span_id], d + 1
            return d

        for s in sorted(spans, key=lambda s: s.start_time):
            pad = "  " * depth(s)
            kind = (s.attributes or {}).get("openinference.span.kind", "-")
            print(f"\n{pad}--- {s.name}  [OI {kind}]  status={s.status.status_code.name}")
            for k, v in sorted((s.attributes or {}).items()):
                if RAW or k.startswith("gen_ai.") or k == "error.type":
                    v = str(v)
                    print(f"{pad}    {k} = {v[:160]}{'…' if len(v) > 160 else ''}")
        return self.inner.export(spans) if self.inner else SpanExportResult.SUCCESS

    def shutdown(self):
        if self.inner:
            self.inner.shutdown()

    def force_flush(self, timeout_millis=30_000):
        return self.inner.force_flush(timeout_millis) if self.inner else True


if DRY_RUN:
    resource, inner = Resource({}), None
else:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    ENTITY, PROJECT = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]
    resource = Resource({"wandb.entity": ENTITY, "wandb.project": PROJECT})
    inner = OTLPSpanExporter(endpoint=AGENTS_URL, headers={"wandb-api-key": os.environ["WANDB_API_KEY"]})

spans = load_spans(TRACE, resource)
printer = Print(inner)
exporter = printer if RAW else OpenInferenceToGenAIExporter(printer)
# One batch (like BatchSpanProcessor); the bridge processes it in end-time order.
exporter.export(spans)
exporter.shutdown()

sessions = sorted({s.attributes["session.id"] for s in spans if "session.id" in s.attributes})
dest = "DRY_RUN: nothing uploaded" if DRY_RUN else f"{ENTITY}/{PROJECT} Agents tab"
print(f"\n[{'RAW' if RAW else 'bridged'}] {len(spans)} spans  session={', '.join(sessions)}  ->  {dest}")
