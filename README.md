# OpenInference → GenAI → Weave Agents

Two files:

- `lang-to-genai.py` — LangGraph agent that emits OpenInference spans and ships them to Weave
- `oi_genai_bridge.py` — wraps an OTel exporter and adds `gen_ai.*` attributes so the Agents tab can render the trace

The Agents endpoint (`/agents/otel/v1/traces`) expects GenAI semantic conventions. OpenInference alone is not enough for that view — that's what the bridge is for.

## Setup

```bash
pip install langgraph langchain langchain-openai openinference-instrumentation-langchain \
            opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

## Run

```bash
python lang-to-genai.py                  # bridged → Agents tab
RAW=1 python lang-to-genai.py            # no bridge (Agents tab stays empty)
TARGET=traces python lang-to-genai.py    # untouched OI → Traces tab
TARGET=both python lang-to-genai.py      # Agents + Traces
DRY_RUN=1 python lang-to-genai.py        # print only, don't upload
```

---

## Flow

```
LangGraph agent.invoke()
        │
        ▼
LangChainInstrumentor  (OpenInference attrs on each span)
        │
        ▼
SimpleSpanProcessor
        │
        ├── TARGET=agents / both
        │         │
        │         ▼
        │   OpenInferenceToGenAIExporter   ← oi_genai_bridge.py
        │   (adds gen_ai.* , keeps OI attrs)
        │         │
        │         ▼
        │   PrintThenForward
        │         │
        │         ▼
        │   POST https://trace.wandb.ai/agents/otel/v1/traces
        │         → Agents tab
        │
        └── TARGET=traces / both
                  │
                  ▼
            PrintThenForward (no bridge)
                  │
                  ▼
            POST https://trace.wandb.ai/otel/v1/traces
                  → Traces / Threads tab
```

`/agents/otel/v1/traces` is the same OTLP HTTP traces ingest as `/otel/v1/traces`, but Weave routes it into the Agents UI and looks for `gen_ai.*` (operation name, conversation id, messages, etc.). Without those attrs, the Agents tab has nothing useful to show.

---

## `lang-to-genai.py` — what each call does

| Call | What it does |
|------|----------------|
| `OTLPSpanExporter(endpoint=..., headers={"wandb-api-key": ...})` | Sends spans to Weave over OTLP/HTTP. `otlp()` returns `None` when `DRY_RUN=1`. |
| `TracerProvider(resource={"wandb.entity", "wandb.project"})` | OTel tracer provider; resource tags tell Weave which entity/project owns the traces. |
| `PrintThenForward(...)` | Custom exporter: prints span attrs (all, or just `gen_ai.*`), then forwards to the real exporter. |
| `OpenInferenceToGenAIExporter(printer, agent_name=...)` | Bridge: maps OpenInference → `gen_ai.*`, then hands spans to `PrintThenForward`. Skipped when `RAW=1`. |
| `provider.add_span_processor(SimpleSpanProcessor(...))` | Exports each span as soon as it ends (not batched), so print order matches end order. |
| `LangChainInstrumentor().instrument(tracer_provider=provider)` | Auto-instruments LangChain/LangGraph with OpenInference attributes. |
| `@tool` / `get_weather` | Dummy tool so the agent produces TOOL spans. |
| `ChatOpenAI(...)` | LLM client pointed at W&B Inference (`api.inference.wandb.ai`). |
| `create_agent(llm, tools=[get_weather])` | Builds the LangGraph agent. |
| `using_session(session_id)` | Sets OpenInference `session.id` on spans in the block → bridge maps it to `gen_ai.conversation.id`. |
| `agent.invoke({"messages": messages})` | Runs one turn; instrumentor creates the span tree (root + LLM + TOOL). |
| `provider.shutdown()` | Flushes any remaining spans and shuts exporters down. |

Env knobs:

- `TARGET` — `agents` (default) / `traces` / `both`
- `RAW=1` — skip the bridge
- `DRY_RUN=1` — print only
- `INFERENCE_MODEL` — model id (default `meta-llama/Llama-3.3-70B-Instruct`)

---

## `oi_genai_bridge.py` — how it works

`OpenInferenceToGenAIExporter` wraps another exporter. On `export()`:

1. Sort spans by end time (children finish before the root with `SimpleSpanProcessor`).
2. For each span, `MessageBridge.convert()` builds extra `gen_ai.*` attrs from OpenInference.
3. Clone the span with those attrs merged in (never overwrites existing `gen_ai.*`, never removes OI attrs).
4. Pass the list to the inner exporter.

### Attribute mapping

| OpenInference | GenAI |
|---------------|--------|
| root span | `gen_ai.operation.name` = `invoke_agent` |
| `openinference.span.kind` = LLM | `gen_ai.operation.name` = `chat` |
| `openinference.span.kind` = TOOL | `gen_ai.operation.name` = `execute_tool` |
| `session.id` | `gen_ai.conversation.id` |
| `agent.name` or `agent_name=` arg | `gen_ai.agent.name` (root) |
| `llm.model_name` | `gen_ai.request.model` |
| `llm.provider` | `gen_ai.provider.name` |
| `llm.token_count.prompt/completion` | `gen_ai.usage.input/output_tokens` |
| `tool.name` | `gen_ai.tool.name` |
| `llm.input/output_messages.*` | `gen_ai.input/output.messages` (LLM spans) |
| `input.value` / `output.value` | `gen_ai.tool.call.arguments` / `result` (TOOL spans) |

### Turn state on the root span

While LLM spans export, the bridge remembers the latest user message and the last model output for that `trace_id`. When the root span exports, those become `gen_ai.input.messages` / `gen_ai.output.messages` on the root — what the Agents UI uses for the turn summary.

Same idea as `otel-collector-config.yaml`, but in-process so you don't need a collector.

---

## Optional: `otel-collector-config.yaml`

Same OI → `gen_ai.*` transform as the Python bridge, but in an OTel Collector. Use this when you already send OTLP to a collector, or when you want the mapping as a separate deploy stage (e.g. a Dockerized collector between your app and Weave) instead of in-process.

Point the collector at this config; it listens on OTLP (4317/4318), adds the GenAI attrs, and exports to `https://trace.wandb.ai/agents/otel/v1/traces`. Needs `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` (and optional `AGENT_NAME`).
