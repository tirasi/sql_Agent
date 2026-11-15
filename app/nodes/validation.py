"""
Node 4: validation
Checks the SQL on the read-only connection. Records every attempt in
attempt_history. If the query tried to WRITE, marks it blocked and sets a
refusal answer -- retrying a DELETE request would just burn attempts.
"""

from app.state.agent_state import AgentState
from app.database.connection import get_readonly_connection
from app.sql.validator import validate_sql, is_write_attempt


def validation_node(state: AgentState) -> dict:
    conn = get_readonly_connection()
    is_valid, error = validate_sql(conn, state.get("sql_query") or "")
    blocked = is_write_attempt(error)

    out = {
        "is_valid": is_valid,
        "validation_error": error,
        "blocked": blocked,
        "attempt_history": [{
            "attempt": state.get("attempts", 0),
            "stage": "validation",
            "sql": state.get("sql_query"),
            "error": error,
        }],
    }
    if blocked:
        out["final_answer"] = (
            "Refused: this agent is read-only. The request would modify the "
            "database, so no query was run."
        )
    return out
