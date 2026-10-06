from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from state import AgentState, make_initial_state
from tool_node import tool_node
from approval import approval_node, HIGH_STAKES_TOOLS
import asyncio
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
import mcp_client
from reasoning import reasoning_node, bind_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from summarize import summarize_node
from hooks import run_pre_hooks

load_dotenv()  # loads GROQ_API_KEY from .env

MAX_ITERATIONS = 8
MAX_MESSAGES = 12            # compact when the history grows beyond this
THREAD_ID = "ticket-001"
DB_PATH = "checkpoints.db"

def _claims_refund_without_tool(state: AgentState) -> bool:
    text = (state["messages"][-1].content or "").lower()
    called = any(getattr(m, "name", None) == "issue_refund" for m in state["messages"])
    nudged = any(isinstance(m, HumanMessage) and "System check" in m.content for m in state["messages"])
    return "refund" in text and not called and not nudged   # nudge at most once


def refund_guard_node(state: AgentState) -> dict:
    return {"messages": [HumanMessage(content=(
        "System check: your reply mentions a refund, but issue_refund was never called. "
        "If a refund is warranted, call issue_refund now with the exact duplicate charge amount. "
        "Otherwise rewrite the reply without promising a refund."
    ))]}

def route_after_reasoning(state: AgentState) -> Literal["tools", "approval", "guard", "end"]:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return "guard" if _claims_refund_without_tool(state) else "end"
    risky = [c for c in calls if c["name"] in HIGH_STAKES_TOOLS]
    # Only valid high-stakes requests go to the human; invalid ones get a 400 from the tool node
    if risky and not any(run_pre_hooks(c["name"], c["args"], state) for c in risky):
        return "approval"
    return "tools"


def route_after_tools(state: AgentState) -> Literal["summarize", "reasoning", "end"]:
    if state["status"] == "done" or state["iterations"] >= MAX_ITERATIONS:
        return "end"
    if len(state["messages"]) > MAX_MESSAGES:
        return "summarize"
    return "reasoning"


builder = StateGraph(AgentState)
builder.add_node("reasoning", reasoning_node)
builder.add_node("tools", tool_node)
builder.add_node("approval", approval_node)
builder.add_node("summarize", summarize_node)
builder.add_node("guard", refund_guard_node)
builder.set_entry_point("reasoning")
builder.add_conditional_edges("reasoning", route_after_reasoning, {"tools": "tools", "approval": "approval", "guard": "guard", "end": END})
builder.add_edge("guard", "reasoning")
builder.add_conditional_edges("tools", route_after_tools, {"summarize": "summarize", "reasoning": "reasoning", "end": END})
builder.add_edge("summarize", "reasoning")
graph = builder.compile(checkpointer=MemorySaver(),interrupt_before=["approval"])

@observe(name="support-ticket-graph", capture_input=False, capture_output=False)
async def run_graph(app, graph_input, config):
    """One traced segment of the run (a start, or a resume after the human's answer)."""
    thread_id = config["configurable"]["thread_id"]
    langfuse = get_client()
    langfuse.update_current_span(
        input={"thread_id": thread_id, "resumed_from_checkpoint": graph_input is None}
    )
    with propagate_attributes(session_id=thread_id, tags=["support-agent"]):
        handler = CallbackHandler()
        result = await app.ainvoke(graph_input, {**config, "callbacks": [handler]})
        # Compact run log on the trace itself, readable at a glance and by the judge
        langfuse.update_current_span(output={
            "status": result.get("status"),
            "iterations": result.get("iterations"),
            "scratchpad": result.get("scratchpad"),
            "errors": result.get("errors"),
            "draft_reply": result.get("draft_reply"),
            "resolution": result.get("resolution"),
            "escalation_reason": result.get("escalation_reason"),
        })
        return result

async def main():
    async with mcp_client.connect("mcp_server.py"):
        schemas = await mcp_client.list_tool_schemas()
        print("Discovered tools:", [s["function"]["name"] for s in schemas])
        bind_mcp_tools(schemas)

        # Checkpoints are written to a SQLite file after every step
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
            app = builder.compile(checkpointer=saver, interrupt_before=["approval"])
            config = {"configurable": {"thread_id": THREAD_ID}}

            snapshot = await app.aget_state(config)
            if not snapshot.values:                  # new thread: start a fresh run
                state = make_initial_state(
                    "I was charged twice for my subscription this month! Please refund the duplicate charge.",
                    "C-1042",
                )
                await run_graph(app, state, config)
            else:
                print("Found saved state on disk, resuming...")
            snapshot = await app.aget_state(config)

            while snapshot.next:
                if snapshot.next == ("approval",):   # paused for the human
                    print("\n=== APPROVAL REQUIRED ===")
                    print("Ticket:", snapshot.values["ticket"])
                    for call in snapshot.values["messages"][-1].tool_calls:
                        if call["name"] in HIGH_STAKES_TOOLS:
                            print("Action:", call["name"], call["args"])

                    answer = ""
                    while answer not in ("approve", "reject"):
                        answer = (await asyncio.to_thread(input, "Type Approve or Reject: ")).strip().lower()

                    decision = "approved" if answer == "approve" else "rejected"
                    await app.aupdate_state(config, {"approval": decision}, as_node="reasoning")
                # otherwise the run was interrupted mid-way (for example by a crash): just continue
                await run_graph(app, None, config)
                snapshot = await app.aget_state(config)

            final = snapshot.values
            for msg in final["messages"]:
                msg.pretty_print()

            print("\n--- Final state ---")
            for key in ["category", "urgency", "summary", "draft_reply", "status", "iterations", "errors"]:
                print(f"{key}: {final[key]}")

    get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())