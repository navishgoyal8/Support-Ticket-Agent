from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage,
)

import reasoning
from state import AgentState

SUMMARIZE_COUNT = 8   # summarize (at most) the first 8 messages


def _render(messages) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Human/System: {m.content}")
        elif isinstance(m, AIMessage):
            for c in m.tool_calls:
                lines.append(f"Agent called tool {c['name']} with {c['args']}")
            if m.content:
                lines.append(f"Agent: {m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result ({m.name}): {m.content}")
    return "\n".join(lines)


async def summarize_node(state: AgentState) -> dict:
    messages = state["messages"]

    # Never leave a tool result without the AI message that requested it
    cut = SUMMARIZE_COUNT
    while cut > 0 and cut < len(messages) and isinstance(messages[cut], ToolMessage):
        cut -= 1
    if cut <= 0:
        return {}

    old = messages[:cut]
    instructions = (
        "Summarize the conversation segment below into ONE paragraph for a customer-support agent "
        "who will continue the work. Keep ticket facts, IDs, amounts, dates, tool results, decisions, "
        "approvals or rejections, and anything still pending. No preamble."
    )
    if state.get("summary"):
        instructions += f"\n\nMerge in this earlier summary:\n{state['summary']}"

    # Plain model call: no tools bound, and its output is not added to the message history
    response = await reasoning.llm.ainvoke([
        SystemMessage(content=instructions),
        HumanMessage(content=_render(old)),
    ])

    return {
        "messages": [RemoveMessage(id=m.id) for m in old],   # delete the old messages from state
        "summary": response.content.strip(),                  # the summary replaces them
        "scratchpad": [f"Compacted {len(old)} messages into a summary"],
    }