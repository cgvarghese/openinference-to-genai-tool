"""
multi-agent-to-genai.py — live LangGraph version of mock_trace.json: SupervisorAgent delegates to
BankingAgent (get_transactions -> transaction retriever) and DisputeAgent (file_dispute).
Real LLM calls, real OpenInference auto-instrumentation, then oi_genai_bridge -> Weave Agents.

Span shape it produces (same kinds as the mock):
  SupervisorAgent [AGENT] -> model [LLM] / delegate [TOOL] -> BankingAgent [AGENT]
    -> get_transactions [TOOL] -> retrieve_transactions [RETRIEVER] ...
  Agent names land in OI `metadata` (via using_metadata), like Capital One's traces.

  pip install langgraph langchain langchain-openai openinference-instrumentation-langchain \
              opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
  export WANDB_API_KEY=... WANDB_ENTITY=... WANDB_PROJECT=...
  python multi-agent-to-genai.py                    # happy path (like mock_trace_success.json)
  SCENARIO=error python multi-agent-to-genai.py     # first get_transactions times out, agent retries (like mock_trace.json)
  RAW=1 / DRY_RUN=1                                  # same as lang-to-genai.py
"""

import json
import os
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openinference.instrumentation import using_metadata, using_session
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from oi_genai_bridge import OpenInferenceToGenAIExporter

KEY = os.environ["WANDB_API_KEY"]
ENTITY = os.getenv("WANDB_ENTITY")
PROJECT = os.getenv("WANDB_PROJECT")
DRY_RUN = os.getenv("DRY_RUN") == "1"
RAW = os.getenv("RAW") == "1"
SCENARIO = os.getenv("SCENARIO", "success")
if SCENARIO not in ("success", "error"):
    raise SystemExit(f"SCENARIO must be success or error (got {SCENARIO!r})")
if not DRY_RUN and not (ENTITY and PROJECT):
    raise SystemExit("WANDB_ENTITY and WANDB_PROJECT are required unless DRY_RUN=1")


class PrintThenForward(SpanExporter):
    """Prints each span's gen_ai.* / error.type attributes (all when RAW), then forwards (unless DRY_RUN)."""

    def __init__(self, inner=None):
        self.inner = inner

    def export(self, spans):
        for s in spans:
            kind = (s.attributes or {}).get("openinference.span.kind", "-")
            print(f"\n--- {s.name}  [OI {kind}]  status={s.status.status_code.name}  root={s.parent is None}")
            for k, v in sorted((s.attributes or {}).items()):
                if RAW or k.startswith("gen_ai.") or k == "error.type":
                    v = str(v)
                    print(f"    {k} = {v[:160]}{'…' if len(v) > 160 else ''}")
        return self.inner.export(spans) if self.inner else SpanExportResult.SUCCESS

    def shutdown(self):
        if self.inner:
            self.inner.shutdown()

    def force_flush(self, timeout_millis=30_000):
        return self.inner.force_flush(timeout_millis) if self.inner else True


AGENTS_URL = "https://trace.wandb.ai/agents/otel/v1/traces"

if DRY_RUN:
    inner = None
else:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    inner = OTLPSpanExporter(endpoint=AGENTS_URL, headers={"wandb-api-key": KEY})

provider = TracerProvider(resource=Resource({"wandb.entity": ENTITY or "", "wandb.project": PROJECT or ""}))
printer = PrintThenForward(inner)
provider.add_span_processor(SimpleSpanProcessor(printer if RAW else OpenInferenceToGenAIExporter(printer)))
LangChainInstrumentor().instrument(tracer_provider=provider)


# --- Fake core-banking data -----------------------------------------------------------------

TRANSACTIONS = [
    {"id": "txn_amzn_8842", "date": "2026-10-02", "merchant": "AMAZON.COM", "amount": 49.99, "mcc": "5942"},
    {"id": "txn_7731", "date": "2026-10-01", "merchant": "WHOLE FOODS MKT", "amount": 82.14, "mcc": "5411"},
    {"id": "txn_6690", "date": "2026-09-29", "merchant": "SHELL OIL 574", "amount": 45.00, "mcc": "5541"},
]


class UpstreamTimeoutError(Exception):
    pass


class TransactionRetriever(BaseRetriever):
    """Scores every transaction against the query by merchant match (stand-in for a vector store)."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = []
        for t in TRANSACTIONS:
            score = 0.98 if t["merchant"].split()[0].split(".")[0].lower() in query.lower() else 0.35
            docs.append(Document(
                id=t["id"],
                page_content=f"{t['date']} | {t['merchant']} | ${t['amount']:.2f} | card ****4821 | status: posted",
                metadata={"merchant": t["merchant"], "amount": t["amount"], "mcc": t["mcc"], "score": score},
            ))
        return sorted(docs, key=lambda d: -d.metadata["score"])


retriever = TransactionRetriever().with_config(run_name="retrieve_transactions")
calls = {"get_transactions": 0}


@tool
def get_transactions(start_date: str, end_date: str, merchant_filter: str = "") -> str:
    """Retrieve account transactions in a date range (YYYY-MM-DD), optionally filtered by merchant."""
    calls["get_transactions"] += 1
    if SCENARIO == "error" and calls["get_transactions"] == 1:
        raise UpstreamTimeoutError("core-banking transactions service timed out after 5000ms")
    docs = retriever.invoke(f"{merchant_filter} purchases, account ****4821, {start_date} to {end_date}")
    hits = [d for d in docs if d.metadata["score"] > 0.9 and start_date <= d.page_content[:10] <= end_date]
    return json.dumps([{"id": d.id, "date": d.page_content[:10], "merchant": d.metadata["merchant"],
                        "amount": d.metadata["amount"], "card": "****4821"} for d in hits])


@tool
def file_dispute(transaction_id: str, reason: str, amount: float | None = None) -> str:
    """Open a dispute for a transaction."""
    return json.dumps({"dispute_id": "DSP-5521", "status": "submitted", "transaction_id": transaction_id})


def surface_timeout(exc: Exception, request) -> str | None:
    # Hand the timeout back to the model (so it can retry) instead of crashing the run.
    if isinstance(exc, UpstreamTimeoutError):
        return f"ERROR: UpstreamTimeoutError: {exc}"
    return None


# --- Agents ---------------------------------------------------------------------------------

llm = ChatOpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=KEY,
    model=os.getenv("INFERENCE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    temperature=0.0,
    max_tokens=1024,
    default_headers={"OpenAI-Project": f"{ENTITY}/{PROJECT}"} if ENTITY and PROJECT else None,
)

# Graph name contains "Agent" -> OpenInference marks its span AGENT (same as Capital One's traces).
sub_agents = {
    "banking_agent": ("BankingAgent", create_agent(
        llm, tools=[get_transactions], name="BankingAgent",
        middleware=[ToolErrorMiddleware(surface_timeout)],
        system_prompt="You are a banking agent. Look up account activity with the get_transactions tool. "
                      "Today is 2026-10-07; search the last 30 days. If the tool times out, retry it once. "
                      "Report the matching transaction id, date and amount.",
    )),
    "dispute_agent": ("DisputeAgent", create_agent(
        llm, tools=[file_dispute], name="DisputeAgent",
        system_prompt="You are a dispute agent. Use the file_dispute tool to open disputes, "
                      "with reason 'unauthorized_transaction'. Report the dispute id.",
    )),
}


@tool
def delegate(agent: str, task: str) -> str:
    """Delegate a task to a named sub-agent ("banking_agent" or "dispute_agent") and return its result."""
    if agent not in sub_agents:
        return f"Unknown agent {agent!r}; use banking_agent or dispute_agent."
    name, sub = sub_agents[agent]
    with using_metadata({"agent.name": name, "delegated_by": "SupervisorAgent"}):
        result = sub.invoke({"messages": [("user", task)]})
    return result["messages"][-1].content


supervisor = create_agent(
    llm, tools=[delegate], name="SupervisorAgent",
    system_prompt="You are a banking supervisor assistant. Delegate account lookups to banking_agent and "
                  "dispute filing to dispute_agent, one at a time, using the delegate tool. "
                  "Only dispute a transaction if the lookup found one. Then answer the customer.",
)

QUESTION = ("I got a notification that I purchased something from Amazon, could you check my transaction "
            "history and verify if that's true? I never purchased anything from Amazon, so if that "
            "transaction exists, please dispute it.")

session_id = f"multi-agent-{uuid.uuid4().hex[:8]}"
with using_session(session_id), using_metadata({"agent.name": "SupervisorAgent", "langgraph.graph": "supervisor"}):
    answer = supervisor.invoke({"messages": [("user", QUESTION)]})["messages"][-1].content
print(f"\n>>> {QUESTION}\n<<< {answer}")

provider.shutdown()
dest = "DRY_RUN: nothing uploaded" if DRY_RUN else f"{ENTITY}/{PROJECT} Agents tab"
print(f"\n[{'RAW (no bridge)' if RAW else 'bridged'}] scenario={SCENARIO}  session={session_id}  "
      f"get_transactions calls={calls['get_transactions']}  ->  {dest}")
