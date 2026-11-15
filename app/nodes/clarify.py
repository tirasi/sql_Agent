"""
Node: clarify  (C9)
Reached when the candidate queries disagree and no reading has a majority.
Instead of silently guessing ("top customers" by count? by spend?), the
user gets the readings with a preview of each answer. Their reply ("2",
"the second one", "by total spend") is matched in context_resolution and
the chosen reading's SQL is reused -- no new LLM call.
"""

from app.state.agent_state import AgentState


def clarify_node(state: AgentState) -> dict:
    opts = state.get("clarification_options") or []
    lines = [f"{i}. {o['interpretation']} -> {o['preview']}" for i, o in enumerate(opts, 1)]
    return {"final_answer": (
        "Your question can be read in more than one way, and the readings give different answers:\n"
        + "\n".join(lines)
        + "\nReply with the number (or rephrase the question) and I'll answer that one."
    )}
