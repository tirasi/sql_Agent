"""
Node 3: sql_generation  (your original "write_sql")
Writes SQL on the first attempt, or a corrected query on retries after a
validation failure -- using a different prompt template for each case.
"""
import os
from app.state.agent_state import AgentState
from app.llm.client import call_llm

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r") as f:
        return f.read()


def sql_generation_node(state: AgentState) -> dict:
    question = state["resolved_input"] or state["user_input"]
    schema_context = state["schema_context"] or ""
    # FIX: this was previously hardcoded to "" in both branches below,
    # which meant the model never saw FK/join relationships between tables.
    join_context = state.get("join_context", "") or ""

    if state.get("attempts", 0) == 0 or not state.get("validation_error"):
        template = _load_prompt("sql_generation.md")
        system_prompt = template.format(
            schema_context=schema_context,
            join_context=join_context,
        )
    else:
        template = _load_prompt("sql_retry.md")
        system_prompt = template.format(
            validation_error=state["validation_error"],
            previous_query=state["sql_query"],
            schema_context=schema_context,
            join_context=join_context,
        )

    raw_sql = call_llm(system_prompt, question, node_name="sql_generation")
    clean_sql = (
        raw_sql.strip()
        .strip(";")
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    return {"sql_query": clean_sql, "attempts": state.get("attempts", 0) + 1}