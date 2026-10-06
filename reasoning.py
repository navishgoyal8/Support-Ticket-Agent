from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from prompts import SYSTEM_PROMPT

load_dotenv()  # loads GROQ_API_KEY from .env

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm

def bind_mcp_tools(schemas: list[dict]) -> None:
    global llm_with_tools
    llm_with_tools = llm.bind_tools(schemas)
    
async def reasoning_node(state: AgentState) -> dict:    
    system = SystemMessage(content=SYSTEM_PROMPT.format(
        category=state.get("category") or "not set",
        urgency=state.get("urgency") or "not set",
        customer_info=state.get("customer_info") or "none",
        kb_results=state.get("kb_results") or "none",
        review_feedback=state.get("review_feedback") or "none",
        summary=state.get("summary") or "none",
    ))

    # First turn: seed the conversation with the ticket
    new_messages = []
    history = state["messages"]
    ticket_msg = HumanMessage(content=f"Customer ID: {state.get('customer_id')}\n\nTicket: {state['ticket']}")
    if not history:
        history = [ticket_msg]
        new_messages.append(ticket_msg)
    elif state.get("summary"):
        history = [ticket_msg] + history     # original ticket message was compacted away

    response = await llm_with_tools.ainvoke([system] + history)
    new_messages.append(response)

    update = {"messages": new_messages, "iterations": state["iterations"] + 1}
    if not response.tool_calls:          # final text -> hand off to review
        update["draft_reply"] = response.content
        update["status"] = "reviewing"
    return update
