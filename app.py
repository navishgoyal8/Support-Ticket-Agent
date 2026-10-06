import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from langfuse import get_client
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()   # local runs only; on the host the variables come from its dashboard

import mcp_client
from graph import builder, run_graph, HIGH_STAKES_TOOLS
from reasoning import bind_mcp_tools
from state import make_initial_state

log = logging.getLogger("uvicorn.error")
HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(HERE, "mcp_server.py")
DB_PATH = os.environ.get("CHECKPOINT_DB", os.path.join(HERE, "checkpoints.db"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch the MCP server, discover tools (tools/list), open the checkpoint DB
    async with mcp_client.connect(SERVER_SCRIPT):
        bind_mcp_tools(await mcp_client.list_tool_schemas())
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
            app.state.agent = builder.compile(checkpointer=saver, interrupt_before=["approval"])
            yield
    get_client().flush()


app = FastAPI(title="Support Ticket Agent", lifespan=lifespan)


class WebhookIn(BaseModel):
    ticket: str
    customer_id: Optional[str] = None
    thread_id: Optional[str] = None


class ApprovalIn(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]


def _auth(key: Optional[str]) -> None:
    expected = os.environ.get("WEBHOOK_API_KEY")
    if not expected:
        raise HTTPException(500, "WEBHOOK_API_KEY is not configured on the server")
    if not secrets.compare_digest(key or "", expected):
        raise HTTPException(401, "Invalid API key")


async def _run(agent, graph_input, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        await run_graph(agent, graph_input, config)
    except Exception as e:
        log.exception("agent run failed")
        raise HTTPException(502, f"Agent run failed: {type(e).__name__}")


async def _result(agent, thread_id: str) -> dict:
    snap = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    values = snap.values
    if snap.next == ("approval",):                      # paused, waiting for a human
        pending = [
            {"tool": c["name"], "args": c["args"]}
            for c in values["messages"][-1].tool_calls
            if c["name"] in HIGH_STAKES_TOOLS
        ]
        return {"thread_id": thread_id, "status": "awaiting_approval", "pending_actions": pending}
    return {
        "thread_id": thread_id,
        "status": "completed" if not snap.next else "incomplete",
        "category": values.get("category"),
        "urgency": values.get("urgency"),
        "resolution": values.get("resolution"),
        "reply": values.get("draft_reply"),
        "escalation_reason": values.get("escalation_reason"),
        "iterations": values.get("iterations"),
        "errors": values.get("errors"),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(body: WebhookIn, request: Request, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    agent = request.app.state.agent
    thread_id = body.thread_id or f"ticket-{uuid.uuid4().hex[:8]}"
    existing = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    if existing.values:
        raise HTTPException(409, "thread_id already exists; use /approve or choose a new thread_id")
    await _run(agent, make_initial_state(body.ticket, body.customer_id), thread_id)
    return await _result(agent, thread_id)


@app.post("/approve")
async def approve(body: ApprovalIn, request: Request, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": body.thread_id}}
    snap = await agent.aget_state(config)
    if snap.next != ("approval",):
        raise HTTPException(409, "No pending approval for this thread_id")
    decision = "approved" if body.decision == "approve" else "rejected"
    await agent.aupdate_state(config, {"approval": decision}, as_node="reasoning")
    await _run(agent, None, body.thread_id)
    return await _result(agent, body.thread_id)