import asyncio
from langchain_core.messages import AIMessage

import mcp_client
from graph import graph
from approval import HIGH_STAKES_TOOLS
import reasoning
from state import make_initial_state

class FakeLLM:
    """Stands in for the Groq model: first asks for a refund, then writes a final reply."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "issue_refund",
                "args": {"customer_id": "C-1042", "amount": 499, "reason": "Duplicate charge on 2026-10-01"},
                "id": "call_test_1",
                "type": "tool_call",
            }])
        return AIMessage(content="Test reply: request handled.")

async def main():
    async with mcp_client.connect("mcp_server.py"):
        reasoning.llm_with_tools = FakeLLM()
        config = {"configurable": {"thread_id": "hitl-test"}}

        state = make_initial_state("Duplicate charge, please refund.", "C-1042")
        state["customer_info"] = {"name": "Demo Customer"}   # the refund pre-hook requires a prior lookup
        await graph.ainvoke(state, config)

        snapshot = await graph.aget_state(config)
        while snapshot.next:        # paused before the "approval" node
            print("\n=== APPROVAL REQUIRED ===")
            print("Ticket:", snapshot.values["ticket"])
            for call in snapshot.values["messages"][-1].tool_calls:
                if call["name"] in HIGH_STAKES_TOOLS:
                    print("Action:", call["name"], call["args"])

            answer = ""
            while answer not in ("approve", "reject"):
                answer = (await asyncio.to_thread(input, "Type Approve or Reject: ")).strip().lower()

            decision = "approved" if answer == "approve" else "rejected"
            await graph.aupdate_state(config, {"approval": decision}, as_node="reasoning")
            await graph.ainvoke(None, config)       # resume from the saved checkpoint
            snapshot = await graph.aget_state(config)

        for msg in snapshot.values["messages"]:
            msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())