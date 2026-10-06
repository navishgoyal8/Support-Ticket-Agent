import json
from langchain_core.messages import ToolMessage

from state import AgentState

import mcp_client
from hooks import run_pre_hooks, run_post_hooks

async def tool_node(state: AgentState) -> dict:
    """Runs every tool the LLM requested and appends the results to messages."""
    last_message = state["messages"][-1]
    tool_messages, notes, errors = [], [], []
    update: dict = {}

    for call in last_message.tool_calls:
        name, args = call["name"], call["args"]

        # PRE-TOOL HOOK: validate before touching the MCP server
        rejection = run_pre_hooks(name, args, state)
        if rejection:
            errors.append(f"{name}: blocked by pre-hook")
            notes.append(f"BLOCKED {name}({args})")
            tool_messages.append(ToolMessage(content=rejection, tool_call_id=call["id"], name=name))
            continue

        try:
            content, is_error = await mcp_client.call_tool(name, args)
            if is_error:
                raise RuntimeError(content)
        except Exception as e:                      # unknown tool, bad args, runtime failure
            content = f"Tool error: {e}"
            errors.append(f"{name}: {e}")
        else:
            # POST-TOOL HOOK: sanitize output before the state or the LLM sees it
            content, hook_notes = run_post_hooks(name, args, content, state)
            notes.extend(hook_notes)
            
            # Copy useful results into the shared state
            if name == "classify_ticket":
                update.update(category=args["category"], urgency=args["urgency"], status="retrieving")
            elif name == "lookup_customer":
                update["customer_info"] = json.loads(content)
            elif name == "search_knowledge_base":
                update["kb_results"] = json.loads(content)
            elif name == "escalate_to_human":
                update.update(resolution="escalated", escalation_reason=args["reason"], status="done")

        tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
        notes.append(f"Ran {name}({args})")

    return {"messages": tool_messages, "scratchpad": notes, "errors": errors, **update}