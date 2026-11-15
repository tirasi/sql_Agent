"""
Node 4: validation
Checks the SQL on a fresh read-only connection (S6). Records every attempt
in attempt_history. Two kinds of query are refused outright, never retried:
  - writes (DELETE/DROP/UPDATE ...)                      -> blocked_reason "write"
  - reads of hidden / uncatalogued tables or columns (S3) -> blocked_reason "restricted"
"""

from app.state.agent_state import AgentState
from app.database.connection import readonly_connection
from app.sql.validator import validate_sql

REFUSALS = {
    "write": ("Refused: this agent is read-only. The request would modify the "
              "database, so no query was run."),
    "restricted": ("Refused: that question needs data this agent is not allowed "
                   "to read, so no query was run."),
}


def validation_node(state: AgentState) -> dict:
    with readonly_connection() as conn:
        is_valid, error = validate_sql(conn, state.get("sql_query") or "")
        denial = conn.last_denial          # "write" | "restricted" | "function" | None

    # a denied FUNCTION (S5) is an ordinary error: the retry loop can pick another one
    reason = denial if (not is_valid and denial in ("write", "restricted")) else None

    out = {
        "is_valid": is_valid,
        "validation_error": error,
        "blocked": bool(reason),
        "blocked_reason": reason,
        "attempt_history": [{
            "attempt": state.get("attempts", 0),
            "stage": "validation",
            "sql": state.get("sql_query"),
            "error": error,
        }],
    }
    if reason:
        out["final_answer"] = REFUSALS[reason]
    return out
