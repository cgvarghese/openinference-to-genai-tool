"""
lang-to-genai.py — LangGraph agent + OpenInference -> oi_genai_bridge -> Weave Agents.

Prints gen_ai.* attributes on every span so you can check the mapping without the UI.

  pip install langgraph langchain langchain-openai openinference-instrumentation-langchain \
              opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
  export WANDB_API_KEY=... WANDB_ENTITY=... WANDB_PROJECT=...
  python lang-to-genai.py                 # bridged: OI + gen_ai.* -> Agents view
  RAW=1 python lang-to-genai.py           # untouched OI, no bridge (Agents view stays empty)
  TARGET=traces python lang-to-genai.py   # untouched OI -> Traces view (/otel/v1/traces)
  TARGET=both python lang-to-genai.py     # bridged -> Agents AND untouched -> Traces
  DRY_RUN=1 ...                           # combine with any of the above: print only, no upload
"""

import os
import uuid

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from openinference.instrumentation import using_session
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from oi_genai_bridge import OpenInferenceToGenAIExporter

KEY = os.environ["WANDB_API_KEY"]
ENTITY = os.environ["WANDB_ENTITY"]
PROJECT = os.environ["WANDB_PROJECT"]
DRY_RUN = os.getenv("DRY_RUN") == "1"
# RAW=1: skip the bridge and send the untouched OpenInference trace (the "before" state;
# Agents tab should stay empty). Prints every attribute instead of just gen_ai.*.
RAW = os.getenv("RAW") == "1"
# TARGET: agents (default) | traces | both. Which Weave endpoint(s) receive the spans.
TARGET = os.getenv("TARGET", "agents")
if TARGET not in ("agents", "traces", "both"):
    raise SystemExit(f"TARGET must be agents, traces or both (got {TARGET!r})")


class PrintThenForward(SpanExporter):
    """Prints each span's attributes (gen_ai.* only, or all when RAW), then forwards (unless DRY_RUN)."""

    def __init__(self, inner=None, show_all=False, label="", quiet=False):
        self.inner = inner
        self.show_all = show_all
        self.label = label
        self.quiet = quiet

    def export(self, spans):
        for s in [] if self.quiet else spans:
            kind = (s.attributes or {}).get("openinference.span.kind", "-")
            print(f"\n--- [{self.label}] {s.name}  [OI {kind}]  root={s.parent is None}")
            for k, v in sorted((s.attributes or {}).items()):
                if self.show_all or k.startswith("gen_ai."):
                    v = str(v)
                    print(f"    {k} = {v[:160]}{'…' if len(v) > 160 else ''}")
        return self.inner.export(spans) if self.inner else SpanExportResult.SUCCESS

    def shutdown(self):
        if self.inner:
            self.inner.shutdown()

    def force_flush(self, timeout_millis=30_000):
        return self.inner.force_flush(timeout_millis) if self.inner else True


AGENTS_URL = "https://trace.wandb.ai/agents/otel/v1/traces"  # -> Agents view
TRACES_URL = "https://trace.wandb.ai/otel/v1/traces"         # -> Traces / Threads view


def otlp(url):
    return None if DRY_RUN else OTLPSpanExporter(endpoint=url, headers={"wandb-api-key": KEY})


provider = TracerProvider(resource=Resource({"wandb.entity": ENTITY, "wandb.project": PROJECT}))

# Simple (not Batch) processors: export each span as it ends, so output prints in order.
# The bridge handles this fine: children end before the root, so turn state is ready in time.
if TARGET in ("agents", "both"):
    printer = PrintThenForward(otlp(AGENTS_URL), show_all=RAW, label="agents")
    exporter = printer if RAW else OpenInferenceToGenAIExporter(printer, agent_name="lang-to-genai-agent")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
if TARGET in ("traces", "both"):
    # Traces view parses OpenInference natively, so always send the untouched spans here.
    # In "both" mode, don't print a second copy.
    printer = PrintThenForward(otlp(TRACES_URL), show_all=True, label="traces", quiet=(TARGET == "both"))
    provider.add_span_processor(SimpleSpanProcessor(printer))
LangChainInstrumentor().instrument(tracer_provider=provider)


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 24C and sunny in {city}."


llm = ChatOpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=KEY,
    model=os.getenv("INFERENCE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    default_headers={"OpenAI-Project": f"{ENTITY}/{PROJECT}"},
)
agent = create_agent(llm, tools=[get_weather])

session_id = f"lang-to-genai-{uuid.uuid4().hex[:8]}"
messages = []
for question in ["What's the weather in Tokyo?", "And in Paris?"]:
    messages.append(("user", question))
    with using_session(session_id):
        messages = agent.invoke({"messages": messages})["messages"]
    print(f"\n>>> {question}\n<<< {messages[-1].content}")

provider.shutdown()
tabs = {"agents": "Agents tab", "traces": "Traces tab", "both": "Agents + Traces tabs"}[TARGET]
mode = "RAW (no bridge)" if RAW else "bridged"
if TARGET == "traces":
    mode = "untouched"
print(f"\n[{mode}] session={session_id}  ->  {ENTITY}/{PROJECT} {tabs}" + ("  (DRY_RUN: nothing uploaded)" if DRY_RUN else ""))