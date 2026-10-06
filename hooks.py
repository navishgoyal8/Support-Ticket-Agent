import json
import re
from typing import Optional

from state import AgentState


def _bad_request(message: str) -> str:
    """Simulated HTTP 400, returned to the LLM as the tool result."""
    return json.dumps({"status": 400, "error": "Bad Request", "message": message})


# ---------------- PRE-TOOL HOOKS ----------------
# Each hook returns None if the call is allowed, or a 400 string if it is rejected.

def _pre_classify_ticket(args: dict, state: AgentState) -> Optional[str]:
    if state.get("category"):
        return _bad_request(
            f"Ticket is already classified as '{state['category']}'. "
            "Do not call classify_ticket again; continue with the next step."
        )


def _pre_lookup_customer(args: dict, state: AgentState) -> Optional[str]:
    cid = args.get("customer_id", "")
    if not re.fullmatch(r"C-\d+", cid):
        return _bad_request("customer_id must look like 'C-1042'.")
    if not state.get("customer_id"):
        return _bad_request("This ticket has no customer_id, so account lookup is not allowed.")
    if cid != state["customer_id"]:
        return _bad_request(
            f"You may only look up the customer who raised this ticket ({state['customer_id']})."
        )


def _pre_search_knowledge_base(args: dict, state: AgentState) -> Optional[str]:
    query = (args.get("query") or "").strip()
    if len(query) < 3:
        return _bad_request("query is too short. Provide at least 3 characters of keywords.")
    if len(query) > 200:
        return _bad_request("query is too long. Use a short, specific query (max 200 characters).")


def _pre_escalate_to_human(args: dict, state: AgentState) -> Optional[str]:
    if len((args.get("reason") or "").strip()) < 10:
        return _bad_request("reason must clearly explain why escalation is needed (min 10 characters).")

def _pre_issue_refund(args: dict, state: AgentState) -> Optional[str]:
    if args.get("customer_id") != state.get("customer_id"):
        return _bad_request("Refunds can only be issued to the customer who raised this ticket.")
    if not state.get("customer_info"):
        return _bad_request("Call lookup_customer first to confirm the charge before refunding.")
    amount = args.get("amount")
    if not isinstance(amount, (int, float)) or not (0 < amount <= 10000):
        return _bad_request("amount must be a number greater than 0 and at most 10000.")
    if len((args.get("reason") or "").strip()) < 10:
        return _bad_request("reason must explain the refund (min 10 characters).")


PRE_HOOKS = {
    "classify_ticket": _pre_classify_ticket,
    "lookup_customer": _pre_lookup_customer,
    "search_knowledge_base": _pre_search_knowledge_base,
    "escalate_to_human": _pre_escalate_to_human,
    "issue_refund": _pre_issue_refund
}

def run_pre_hooks(name: str, args: dict, state: AgentState) -> Optional[str]:
    hook = PRE_HOOKS.get(name)
    return hook(args, state) if hook else None


# ---------------- POST-TOOL HOOKS ----------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
PHONE_RE = re.compile(r"\b\d{10}\b")


def _redact(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = CARD_RE.sub("[REDACTED_CARD]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def run_post_hooks(name: str, args: dict, output: str, state: AgentState) -> tuple[str, list[str]]:
    """Sanitizes a tool's output. Returns (clean_output, audit_notes)."""
    notes = []
    cleaned = _redact(output)
    if cleaned != output:
        notes.append(f"post-hook: redacted sensitive data from {name} output")
    return cleaned, notes


if __name__ == "__main__":
    # Quick self-test, no LLM needed
    state = {"customer_id": "C-1042", "category": None}
    print(run_pre_hooks("lookup_customer", {"customer_id": "C-9999"}, state))   # 400
    print(run_pre_hooks("lookup_customer", {"customer_id": "C-1042"}, state))   # None (allowed)
    print(run_pre_hooks("escalate_to_human", {"reason": "bad"}, state))         # 400
    print(run_post_hooks("lookup_customer", {}, '{"email": "a@b.com", "card": "4111 1111 1111 1111"}', state))