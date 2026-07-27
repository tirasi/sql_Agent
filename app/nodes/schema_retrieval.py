"""
Node 2: schema_retrieval  (your original "search_agent")
Picks which table(s) are relevant to the question, then builds the full
schema + join context string that sql_generation will use.
"""

from app.state.agent_state import AgentState
from app.llm.client import call_llm
from app.schema import retriever
from app.schema.relationship_resolver import get_join_context


def schema_retrieval_node(state: AgentState) -> dict:
    question = state["resolved_input"] or state["user_input"]

    system_prompt = (
        "You are a database schema router. Given the table catalog below, "
        "respond with ONLY a JSON array of relevant table names for this question.\n\n"
        f"{retriever.get_table_catalog_summary()}"
    )
    raw_output = call_llm(system_prompt, question, node_name="schema_retrieval")

    import json
    try:
        tables = json.loads(raw_output.strip().replace("```json", "").replace("```", ""))
    except (json.JSONDecodeError, ValueError):
        tables = retriever.get_all_table_names()  # safe fallback: consider everything

    schema_context = retriever.get_full_schema_context(tables)
    join_context = get_join_context(tables)

    return {
        "relevant_tables": tables,
        "schema_context": schema_context + "\n\n" + join_context,
    }
