"""
Node 3: sql_generation
Writes SQL on the first attempt, or a corrected query on retries -- using
the error from EITHER validation or execution, plus the previous query.
"""
import os
import re
from app.state.agent_state import AgentState
from app.llm.client import call_llm

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r") as f:
        return f.read()


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and chatter like 'Here is the query:'.
    Fences are removed FIRST, then the trailing ';' (the old order left the
    ';' behind when the model wrapped SQL in ```sql ... ```)."""
    text = raw.replace("```sql", "").replace("```", "").strip()
    m = re.search(r"(?is)\b(with|select)\b.*", text)
    if m:  # keep from the first SELECT/WITH onward
        text = m.group(0)
    # if there is no SELECT (e.g. model wrote DELETE), keep it as-is so the
    # read-only validator sees it and blocks it
    return text.strip().rstrip(";").strip()


def sql_generation_node(state: AgentState) -> dict:
    question = state.get("resolved_input") or state["user_input"]
    schema_context = state.get("schema_context") or ""
    # FIX: join_context is now its own state field (it used to be glued onto
    # schema_context, so this lookup always returned "" and the prompt
    # showed an empty "Join paths:" section).
    join_context = state.get("join_context") or "(single table -- no joins needed)"
    last_error = state.get("validation_error") or state.get("execution_error")

    if state.get("attempts", 0) == 0 or not last_error:
        system_prompt = _load_prompt("sql_generation.md").format(
            schema_context=schema_context,
            join_context=join_context,
        )
    else:
        system_prompt = _load_prompt("sql_retry.md").format(
            error=last_error,
            previous_query=state.get("sql_query"),
            schema_context=schema_context,
            join_context=join_context,
        )

    raw_sql = call_llm(system_prompt, question, node_name="sql_generation")
    return {
        "sql_query": _clean_sql(raw_sql),
        "attempts": state.get("attempts", 0) + 1,
        "execution_error": None,  # clear stale error from the previous attempt
    }
