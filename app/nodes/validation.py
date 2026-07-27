"""
Node 4: validation
Thin wrapper around app.sql.validator -- kept as its own node (rather than
folded into sql_generation) so the graph's conditional retry edge has a
clean, single node to route on.
"""

from app.state.agent_state import AgentState
from app.database.connection import get_connection
from app.sql.validator import validate_sql


def validation_node(state: AgentState) -> dict:
    conn = get_connection()
    is_valid, error = validate_sql(conn, state["sql_query"])
    return {"is_valid": is_valid, "validation_error": error}
