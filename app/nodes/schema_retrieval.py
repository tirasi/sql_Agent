"""
Node 2: schema_retrieval
Picks relevant tables, then builds schema + join context for sql_generation.

The final table set is the UNION of (recall over precision -- an extra table
costs a few prompt tokens; a missing table makes correct SQL impossible):
  1. tables the router LLM picked (filtered to real table names)
  2. tables named in the question, by proper word forms (C6):
     "orders"/"ordered" -> orders, but "in order of" is ignored and
     "status" no longer becomes "statu"
  3. tables whose stored VALUES appear in the question ("Notebook" -> orders)
  4. tables a metric needs ("sales" = SUM(orders.amount) -> orders)      (C4)
  5. on a follow-up question, the tables of the previous turn            (C8)
  6. bridge tables on the FK path between them
On big schemas only the best-matching ROUTER_MAX_TABLES tables are shown to
the router (C10). `table_selection` records WHY each table was picked.
"""

import json
import re
from app.state.agent_state import AgentState
from app.llm.client import call_llm
from app.config import ROUTER_MAX_TABLES
from app.schema import retriever
from app.schema.relationship_resolver import get_join_context, connect_tables
from app.schema.value_index import find_value_mentions


def _parse_router_output(raw: str, known: list[str]) -> list[str] | None:
    """Returns valid table names, or None if the output couldn't be parsed."""
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        picked = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    lower = {t.lower(): t for t in known}
    return [lower[p.lower()] for p in picked if isinstance(p, str) and p.lower() in lower]


def schema_retrieval_node(state: AgentState) -> dict:
    question = state.get("resolved_input") or state["user_input"]
    known = retriever.get_all_table_names()
    reasons: dict[str, list[str]] = {}

    shortlist = retriever.top_tables(question, ROUTER_MAX_TABLES)
    system_prompt = (
        "You are a database schema router. Given the table catalog below, "
        "respond with ONLY a JSON array of relevant table names for this question.\n\n"
        f"{retriever.get_table_catalog_summary(shortlist)}"
    )
    llm_tables = _parse_router_output(
        call_llm(system_prompt, question, node_name="schema_retrieval"), known)
    if llm_tables is None:
        for t in shortlist:
            reasons.setdefault(t, []).append("router output unparseable -> shortlist")
    else:
        for t in llm_tables:
            reasons.setdefault(t, []).append("router LLM")

    for t in retriever.tables_named(question, known):
        reasons.setdefault(t, []).append("table named in question")

    value_hits = find_value_mentions(question)
    for table, col, value in value_hits:
        if table in known:
            reasons.setdefault(table, []).append(f"value '{value}' in {table}.{col}")

    for m in state.get("metric_hints") or []:
        for t in m["tables"]:
            if t in known:
                reasons.setdefault(t, []).append(f"metric '{m['term']}'")

    history = state.get("history") or []
    if state.get("is_follow_up") and history:
        for t in history[-1].get("relevant_tables") or []:
            if t in known:
                reasons.setdefault(t, []).append("used by the previous question (follow-up)")

    if not reasons:
        for t in shortlist:
            reasons[t] = ["nothing matched -> shortlist"]

    tables = [t for t in known if t in reasons]  # stable order
    for t in connect_tables(tables):
        reasons.setdefault(t, []).append("FK bridge")
        tables.append(t)

    schema_context = retriever.get_full_schema_context(tables)
    if value_hits:
        schema_context += "\n\nVALUES MENTIONED IN THE QUESTION (exact stored values):\n" + "\n".join(
            f"  - '{v}' is stored in {t}.{c}" for t, c, v in value_hits)

    return {
        "relevant_tables": tables,
        "table_selection": reasons,
        "schema_context": schema_context,
        "join_context": get_join_context(tables),
    }
