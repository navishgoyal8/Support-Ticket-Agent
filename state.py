from typing import TypedDict, Annotated, Literal, Optional
import operator
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    
    ticket: str                                   
    customer_id: Optional[str]                   
   
    messages: Annotated[list, add_messages]
 
    category: Optional[Literal["billing", "technical", "account", "general"]]
    urgency: Optional[Literal["low", "medium", "high"]]
 
    kb_results: list[dict]                        
    customer_info: dict                           
    scratchpad: Annotated[list[str], operator.add]

    draft_reply: Optional[str]
    confidence: float                             
    review_feedback: Optional[str]                

    resolution: Optional[Literal["replied", "escalated"]]
    approval: Optional[Literal["approved", "rejected"]]   # set by the human at the HITL pause
    summary: str
    final_reply: Optional[str]
    escalation_reason: Optional[str]
 
    status: Literal["classifying", "retrieving", "drafting", "reviewing", "done", "failed"]
    iterations: int
    errors: Annotated[list[str], operator.add]

def make_initial_state(ticket: str, customer_id: str | None = None) -> AgentState:
    return {
        "ticket": ticket, "customer_id": customer_id, "messages": [],
        "category": None, "urgency": None, "kb_results": [], "customer_info": {},
        "scratchpad": [], "draft_reply": None, "confidence": 0.0,
        "review_feedback": None, "resolution": None, "final_reply": None,
        "escalation_reason": None,"approval": None,"summary": "", "status": "classifying", "iterations": 0, "errors": [],
    }