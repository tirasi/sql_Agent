"""
Node 5: execution -- runs the validated SQL on the read-only connection.
Errors here (rare in SQLite after EXPLAIN passed, but possible) are logged
and routed back to sql_generation by the graph, just like validation errors.
"""

from app.state.agent_state import AgentState
from app.database.connection import get_readonly_connection
from app.sql.executor import execute_sql


def execution_node(state: AgentState) -> dict:
    conn = get_readonly_connection()
    rows, columns, error = execute_sql(conn, state["sql_query"])
    out = {"rows": rows, "columns": columns, "execution_error": error}
    if error:
        out["attempt_history"] = [{
            "attempt": state.get("attempts", 0),
            "stage": "execution",
            "sql": state.get("sql_query"),
            "error": error,
        }]
    return out
