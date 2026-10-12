# OpenInference → GenAI → Weave Agents

Weave's Agents tab (`/agents/otel/v1/traces`) only understands GenAI semantic conventions, so OpenInference traces show up empty there. `oi_genai_bridge.py` fixes that by adding the matching `gen_ai.*` attributes to each span before export.

The other scripts generate traces to test it with:

- `lang-to-genai.py`: a simple LangGraph agent with one tool
- `multi-agent-to-genai.py`: a supervisor agent that hands work to two sub-agents (banking and disputes) and includes a live inference test call
- `mock-trace-to-genai.py`: replays the sample traces in `test-spans/`, no LLM needed

---



## Setup

```bash
pip install langgraph langchain langchain-openai openinference-instrumentation-langchain \
            opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

---

## Run

Add `DRY_RUN=1` to any command to print the spans without uploading them.

### `lang-to-genai.py`: single agent

Makes real LLM calls through W&B Inference.

```bash
python lang-to-genai.py                  # send bridged spans to the Agents tab
RAW=1 python lang-to-genai.py            # send without the bridge (Agents tab stays empty)
TARGET=traces python lang-to-genai.py    # send the original spans to the Traces tab instead
TARGET=both python lang-to-genai.py      # send to both tabs
```



### `multi-agent-to-genai.py`: supervisor with sub-agents

Also uses W&B Inference unless you set `LLM=scripted`.

```bash
python multi-agent-to-genai.py                   # transaction lookup works on the first try
SCENARIO=error python multi-agent-to-genai.py    # lookup times out once, then the agent retries
LLM=scripted python multi-agent-to-genai.py      # fixed fake model responses, no Inference key needed
```



### `mock-trace-to-genai.py`: replay a saved trace

No LLM calls. Each run gets a new trace and session id, so it shows up as a new conversation.

```bash
python mock-trace-to-genai.py                    # replay test-spans/mock_trace_success.json
SCENARIO=error python mock-trace-to-genai.py     # replay test-spans/mock_trace.json (timeout + retry)
TRACE=my_trace.json python mock-trace-to-genai.py  # replay your own file
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


| Call                                                             | What it does                                                                                           |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `OTLPSpanExporter(endpoint=..., headers={"wandb-api-key": ...})` | Sends spans to Weave over OTLP/HTTP. `otlp()` returns `None` when `DRY_RUN=1`.                         |
| `TracerProvider(resource={"wandb.entity", "wandb.project"})`     | OTel tracer provider; resource tags tell Weave which entity/project owns the traces.                   |
| `PrintThenForward(...)`                                          | Custom exporter: prints span attrs (all, or just `gen_ai.*`), then forwards to the real exporter.      |
| `OpenInferenceToGenAIExporter(printer, agent_name=...)`          | Bridge: maps OpenInference → `gen_ai.*`, then hands spans to `PrintThenForward`. Skipped when `RAW=1`. |
| `provider.add_span_processor(SimpleSpanProcessor(...))`          | Exports each span as soon as it ends (not batched), so print order matches end order.                  |
| `LangChainInstrumentor().instrument(tracer_provider=provider)`   | Auto-instruments LangChain/LangGraph with OpenInference attributes.                                    |
| `@tool` / `get_weather`                                          | Dummy tool so the agent produces TOOL spans.                                                           |
| `ChatOpenAI(...)`                                                | LLM client pointed at W&B Inference (`api.inference.wandb.ai`).                                        |
| `create_agent(llm, tools=[get_weather])`                         | Builds the LangGraph agent.                                                                            |
| `using_session(session_id)`                                      | Sets OpenInference `session.id` on spans in the block → bridge maps it to `gen_ai.conversation.id`.    |
| `agent.invoke({"messages": messages})`                           | Runs one turn; instrumentor creates the span tree (root + LLM + TOOL).                                 |
| `provider.shutdown()`                                            | Flushes any remaining spans and shuts exporters down.                                                  |


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


| OpenInference                                                                          | GenAI                                                                                                      |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| root span / `openinference.span.kind` = AGENT                                          | `gen_ai.operation.name` = `invoke_agent` (sub-agents too)                                                  |
| `openinference.span.kind` = LLM                                                        | `gen_ai.operation.name` = `chat`                                                                           |
| `openinference.span.kind` = TOOL                                                       | `gen_ai.operation.name` = `execute_tool`, `gen_ai.tool.type` = `function`                                  |
| `openinference.span.kind` = RETRIEVER                                                  | `gen_ai.operation.name` = `execute_tool`, `gen_ai.tool.type` = `datastore`, span name → `gen_ai.tool.name` |
| `session.id`                                                                           | `gen_ai.conversation.id`                                                                                   |
| `agent.name`, `metadata["agent.name"]`, `agent_name=` arg (root), or span name (AGENT) | `gen_ai.agent.name` (agent spans)                                                                          |
| `llm.model_name`                                                                       | `gen_ai.request.model`                                                                                     |
| `llm.provider` (or `llm.system`)                                                       | `gen_ai.provider.name`                                                                                     |
| `llm.invocation_parameters`                                                            | `gen_ai.request.temperature` / `max_tokens` / `top_p` / …                                                  |
| `llm.token_count.prompt/completion`                                                    | `gen_ai.usage.input/output_tokens`                                                                         |
| `tool.name` / `tool.description`                                                       | `gen_ai.tool.name` / `gen_ai.tool.description`                                                             |
| exception event / ERROR status                                                         | `error.type` (e.g. `UpstreamTimeoutError`)                                                                 |
| `llm.input/output_messages.*`                                                          | `gen_ai.input/output.messages` (LLM spans)                                                                 |
| `llm.tools.N.tool.json_schema`                                                         | `gen_ai.tool.definitions` (LLM spans)                                                                      |
| `input.value` / `output.value`                                                         | `gen_ai.tool.call.arguments` / `result` (TOOL spans)                                                       |
| `input.value` / `retrieval.documents.*`                                                | `gen_ai.tool.call.arguments` / `result` as `[{id, content, score, metadata}]` (RETRIEVER spans)            |
| `input.value` / `output.value`                                                         | `gen_ai.input/output.messages` (AGENT + root spans)                                                        |




### Agent messages

Each agent span (the root and any AGENT sub-agent) gets `gen_ai.input.messages` / `gen_ai.output.messages` from its own `input.value` / `output.value`: the user's question and the final answer. This is the turn summary in the Agents tab. If the root has neither, the bridge uses the first user message and last model reply from that trace's LLM spans instead.

Other CHAIN spans (LangGraph nodes and loops) don't get an operation name. They keep their OI attributes and `gen_ai.conversation.id`.

---



## Optional: `otel-collector-config.yaml`

Same OI → `gen_ai.*` transform as the Python bridge, but in an OTel Collector. Use this when you already send OTLP to a collector, or when you want the mapping as a separate deploy stage (e.g. a Dockerized collector between your app and Weave) instead of in-process.

Point the collector at this config; it listens on OTLP (4317/4318), adds the GenAI attrs, and exports to `https://trace.wandb.ai/agents/otel/v1/traces`. Needs `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` (and optional `AGENT_NAME`).