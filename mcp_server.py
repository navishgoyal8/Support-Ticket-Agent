import json
from typing import Literal
from mcp.server.fastmcp import FastMCP
import uuid

mcp = FastMCP("support-tools")

CUSTOMERS = {
    "C-1042": {
        "name": "Demo Customer",
        "email": "demo.customer@example.com",
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
        "content": "Duplicate charges are refunded in full within 5-7 business days. A support agent must issue the refund after the duplicate is confirmed.",    },
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


@mcp.tool()
def classify_ticket(
    category: Literal["billing", "technical", "account", "general"],
    urgency: Literal["low", "medium", "high"],
) -> str:
    """Record the category and urgency of the ticket. Call this first, once."""
    return f"Ticket classified as {category} with {urgency} urgency."


@mcp.tool()
def lookup_customer(customer_id: str) -> str:
    """Fetch account, plan and order details for a customer."""
    data = CUSTOMERS.get(customer_id, {"error": f"No customer found with id {customer_id}"})
    return json.dumps(data)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search help articles and policies. Use a short, specific query."""
    words = set(query.lower().split())
    scored = [(len(words & set(a["keywords"])), a) for a in KB_ARTICLES]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: s[0], reverse=True)
    return json.dumps([{"title": a["title"], "content": a["content"]} for _, a in scored[:2]])


@mcp.tool()
def escalate_to_human(reason: str) -> str:
    """Hand the ticket to a human agent. Use when you cannot resolve it confidently."""
    return f"Ticket escalated to a human agent. Reason: {reason}"

@mcp.tool()
def issue_refund(customer_id: str, amount: float, reason: str) -> str:
    """Issue a refund to a customer. HIGH-STAKES: a human approves every refund before it runs.
    Only use after lookup_customer confirms a duplicate or incorrect charge."""
    return json.dumps({
        "status": "refund_issued",
        "refund_id": f"RF-{uuid.uuid4().hex[:6].upper()}",
        "customer_id": customer_id,
        "amount": amount,
        "reason": reason,
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")