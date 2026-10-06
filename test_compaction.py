import asyncio
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

import reasoning
import summarize
from state import make_initial_state


class FakeSummarizer:
    async def ainvoke(self, messages):
        print("--- text sent to the summarizer ---")
        print(messages[-1].content)
        return AIMessage(content="Customer C-1042 reported a duplicate charge; KB searched several times (fake summary).")


async def main():
    reasoning.llm = FakeSummarizer()

    msgs = [HumanMessage(content="Customer ID: C-1042\n\nTicket: Duplicate charge")]
    for i in range(5):
        msgs.append(AIMessage(content="", tool_calls=[{
            "name": "search_knowledge_base", "args": {"query": f"q{i}"}, "id": f"c{i}", "type": "tool_call"}]))
        msgs.append(ToolMessage(content=f"result {i}", tool_call_id=f"c{i}", name="search_knowledge_base"))

    state = make_initial_state("Duplicate charge", "C-1042")
    state["messages"] = add_messages([], msgs)        # assigns ids, as the real graph does
    print("Before:", len(state["messages"]), "messages")

    update = await summarize.summarize_node(state)
    state["messages"] = add_messages(state["messages"], update["messages"])

    print("\nAfter:", len(state["messages"]), "messages:", [type(m).__name__ for m in state["messages"]])
    print("Summary:", update["summary"])


asyncio.run(main())