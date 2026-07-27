"""
Node 5: response_generation
New vs. your original script: turns raw rows into a readable answer instead
of leaving the caller to print tuples. This is the node most worth
upgrading to a real LLM provider first if you're prioritizing -- it's what
users actually read.
"""

import os
from app.state.agent_state import AgentState
from app.llm.client import call_llm

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r") as f:
        return f.read()


def response_generation_node(state: AgentState) -> dict:
    if state.get("execution_error"):
        return {"final_answer": f"I couldn't get an answer -- the query failed: {state['execution_error']}"}

    template = _load_prompt("response_generation.md")
    prompt = template.format(
        user_input=state["user_input"],
        columns=state["columns"],
        rows=state["rows"],
    )
    answer = call_llm("You summarize query results in plain language.", prompt, node_name="response_generation")
    return {"final_answer": answer}
