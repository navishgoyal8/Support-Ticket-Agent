from typing import Literal
from langchain_core.tools import tool
 
# --- Mock data (swap for a real DB / API / vector search later) ---
CUSTOMERS = {
    "C-1042": {
        "name": "Demo Customer",
        "plan": "Pro Monthly",
        "price": 499,
        "recent_charges": [
            {"date": "2026-10-01", "amount": 499, "status": "paid"},
            {"date": "2026-10-01", "amount": 499, "status": "paid"},
        ],
    },
}

KB_ARTICLES = [
    {
        "title": "Refunds for duplicate charges",
        "keywords": ["refund", "duplicate", "double", "charge", "charged", "twice", "billing"],
        "content": "Duplicate charges are refunded in full within 5-7 business days. No customer action is needed once the duplicate is confirmed.",
    },
    {
        "title": "Resetting your password",
        "keywords": ["password", "reset", "login", "forgot", "account"],
        "content": "Use 'Forgot password' on the login page. A reset link is emailed and expires in 30 minutes.",
    },
    {
        "title": "Cancelling a subscription",
        "keywords": ["cancel", "subscription", "plan", "billing"],
        "content": "Go to Settings > Billing > Cancel plan. Access continues until the end of the billing period.",
    },
]

@tool
def classify_ticket(
    category: Literal["billing", "technical", "account", "general"],
    urgency: Literal["low", "medium", "high"],
) -> str:
    """Record the category and urgency of the ticket. Call this first, once."""

    return f"Ticket classified as {category} with {urgency} urgency."
 
@tool
def lookup_customer(customer_id: str) -> dict:
    """Fetch account, plan and order details for a customer."""

    return CUSTOMERS.get(customer_id, {"error": f"No customer found with id {customer_id}"})
 
@tool
def search_knowledge_base(query: str) -> list:
    """Search help articles and policies. Use a short, specific query."""

    words = set(query.lower().split())
    scored = [(len(words & set(a["keywords"])), a) for a in KB_ARTICLES]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: s[0], reverse=True)
    return [{"title": a["title"], "content": a["content"]} for _, a in scored[:2]]
 
@tool
def escalate_to_human(reason: str) -> str:
    """Hand the ticket to a human agent. Use when you cannot resolve it confidently."""

    return f"Ticket escalated to a human agent. Reason: {reason}"

@tool
def escalate_to_human(reason: str) -> str:
    """Hand the ticket to a human agent. Use when you cannot resolve it confidently."""
    return f"Ticket escalated to a human agent. Reason: {reason}"

TOOLS = [classify_ticket, lookup_customer, search_knowledge_base, escalate_to_human]