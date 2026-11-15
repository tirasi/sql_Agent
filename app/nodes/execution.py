"""
Node 5: execution -- runs the validated SQL on a fresh read-only connection
(S6) with a timeout (S1) and a row cap (S2). Errors are logged and routed
back to sql_generation by the graph, just like validation errors.
"""

from app.state.agent_state import AgentState
from app.database.connection import readonly_connection
from app.sql.executor import execute_sql


def execution_node(state: AgentState) -> dict:
    with readonly_connection() as conn:
        rows, columns, error, truncated = execute_sql(conn, state["sql_query"])
    out = {"rows": rows, "columns": columns, "execution_error": error, "truncated": truncated}
    if error:
        out["attempt_history"] = [{
            "attempt": state.get("attempts", 0),
            "stage": "execution",
            "sql": state.get("sql_query"),
            "error": error,
        }]
    return out
