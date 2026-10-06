from langchain_core.messages import ToolMessage
from langgraph.types import Command

from state import AgentState

HIGH_STAKES_TOOLS = {"issue_refund"}


def approval_node(state: AgentState) -> Command:
    calls = state["messages"][-1].tool_calls

    # The graph was paused BEFORE this node; the human's answer is in state.
    # Anything other than "approved" is treated as a rejection (deny by default).
    if state.get("approval") == "approved":
        return Command(goto="tools", update={
            "approval": None,
            "scratchpad": ["HITL: human approved high-stakes action"],
        })

    # Rejected: every tool call in the AI message needs an answer, or the LLM API errors
    replies = []
    for c in calls:
        if c["name"] in HIGH_STAKES_TOOLS:
            text = ("Rejected by human reviewer. Do not retry this action. "
                    "Call escalate_to_human or tell the customer the case is under review.")
        else:
            text = "Not executed because another action in this step was rejected. Call it again if still needed."
        replies.append(ToolMessage(content=text, tool_call_id=c["id"], name=c["name"]))

    return Command(goto="reasoning", update={
        "approval": None,
        "messages": replies,
        "scratchpad": ["HITL: human rejected high-stakes action"],
    })