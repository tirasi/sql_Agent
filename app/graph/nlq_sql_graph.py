from langgraph.graph import StateGraph, START, END

from app.state.agent_state import AgentState
from app.nodes.context_resolution import context_resolution_node
from app.nodes.schema_retrieval import schema_retrieval_node
from app.nodes.sql_generation import sql_generation_node
from app.nodes.validation import validation_node
from app.nodes.execution import execution_node
from app.nodes.response_generation import response_generation_node
from app.config import MAX_SQL_ATTEMPTS


def give_up_node(state: AgentState) -> dict:
    """Out of retries. Previously the graph just hit END with
    final_answer=None, so the user got nothing; now they get the reason."""
    last = state.get("execution_error") or state.get("validation_error") or "unknown error"
    return {"final_answer": (
        f"I couldn't produce a working query after {state.get('attempts', 0)} attempts. "
        f"Last error: {last}"
    )}


def _route_after_validation(state: AgentState) -> str:
    if state.get("blocked"):
        return END                     # write attempt: refuse, don't retry
    if state["is_valid"]:
        return "execution"
    if state["attempts"] >= MAX_SQL_ATTEMPTS:
        return "give_up"
    return "sql_generation"


def _route_after_execution(state: AgentState) -> str:
    if not state.get("execution_error"):
        return "response_generation"
    if state["attempts"] >= MAX_SQL_ATTEMPTS:
        return "give_up"
    return "sql_generation"            # NEW: runtime errors get a retry too


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("context_resolution", context_resolution_node)
    graph.add_node("schema_retrieval", schema_retrieval_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("validation", validation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("response_generation", response_generation_node)
    graph.add_node("give_up", give_up_node)

    graph.add_edge(START, "context_resolution")
    graph.add_edge("context_resolution", "schema_retrieval")
    graph.add_edge("schema_retrieval", "sql_generation")
    graph.add_edge("sql_generation", "validation")
    graph.add_conditional_edges(
        "validation", _route_after_validation,
        {"execution": "execution", "sql_generation": "sql_generation",
         "give_up": "give_up", END: END},
    )
    graph.add_conditional_edges(
        "execution", _route_after_execution,
        {"response_generation": "response_generation",
         "sql_generation": "sql_generation", "give_up": "give_up"},
    )
    graph.add_edge("response_generation", END)
    graph.add_edge("give_up", END)

    return graph.compile()
