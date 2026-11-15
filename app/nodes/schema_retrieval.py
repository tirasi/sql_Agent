"""
Node 2: schema_retrieval
Picks relevant tables, then builds schema + join context for sql_generation.

FIX: the router LLM alone missed tables (picked only `customers` for "How
many customers ordered notebooks?"). The final table set is now the UNION of:
  1. tables the LLM picked (filtered to real table names -- no hallucinations)
  2. tables named in the question ("orders", "ordered", "customers")
  3. tables whose stored VALUES appear in the question ("Notebook" -> orders)
  4. bridge tables on the FK path between them
Recall over precision: an extra table costs a few prompt tokens; a missing
table makes correct SQL impossible.
`table_selection` records WHY each table was picked (visible in the trace).
"""

import json
import re
from app.state.agent_state import AgentState
from app.llm.client import call_llm
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


def _tables_named(question: str, known: list[str]) -> list[str]:
    # "orders" -> stem "order" -> matches order / orders / ordered
    return [t for t in known
            if re.search(r"\b" + re.escape(t.rstrip("s")) + r"\w*", question, re.IGNORECASE)]


def schema_retrieval_node(state: AgentState) -> dict:
    question = state.get("resolved_input") or state["user_input"]
    known = retriever.get_all_table_names()
    reasons: dict[str, list[str]] = {}

    system_prompt = (
        "You are a database schema router. Given the table catalog below, "
        "respond with ONLY a JSON array of relevant table names for this question.\n\n"
        f"{retriever.get_table_catalog_summary()}"
    )
    llm_tables = _parse_router_output(
        call_llm(system_prompt, question, node_name="schema_retrieval"), known)
    if llm_tables is None:
        for t in known:
            reasons.setdefault(t, []).append("router output unparseable -> all tables")
    else:
        for t in llm_tables:
            reasons.setdefault(t, []).append("router LLM")

    for t in _tables_named(question, known):
        reasons.setdefault(t, []).append("table named in question")

    value_hits = find_value_mentions(question)
    for table, col, value in value_hits:
        if table in known:
            reasons.setdefault(table, []).append(f"value '{value}' in {table}.{col}")

    if not reasons:
        for t in known:
            reasons[t] = ["nothing matched -> all tables"]

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
