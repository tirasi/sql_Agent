"""
Node 1: context_resolution
Normalizes casual user phrasing (e.g. "notebooks" -> "Notebook") using the
synonym map, BEFORE table/SQL generation see the question. Doing this here
(not inline in sql_generation) means every downstream node works with
already-clean terminology, and you can inspect/debug this step in isolation.
"""

from app.state.agent_state import AgentState
from app.schema.retriever import get_synonyms


def context_resolution_node(state: AgentState) -> dict:
    text = state["user_input"]
    synonyms = get_synonyms()
    resolved = text
    applied = {}
    for casual_term, canonical_term in synonyms.items():
        if casual_term.lower() in resolved.lower():
            # simple case-insensitive replace; good enough for a dummy demo,
            # swap for a real LLM call (via app.llm.client.call_llm) if you
            # need this to handle phrasing the dict doesn't literally contain
            resolved = resolved.replace(casual_term, canonical_term)
            applied[casual_term] = canonical_term

    return {"resolved_input": resolved, "detected_synonyms": applied}
