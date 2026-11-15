"""
Node 4: validation
Checks EVERY candidate query (C1) on a fresh read-only connection (S6):
  1. SQLite parses it (EXPLAIN) under the read-only authorizer;
  1b. if it fails with an alias mix-up ("no such column: T2.x" where x is in
      the table aliased T1), it is repaired in code and re-checked -- no
      LLM retry needed (app/sql/repair.py);
  1c. if it is valid, common LOGIC mistakes are rewritten in code
      (COUNT(condition), integer division, ranking a customer by one child
      row, grouping by a full date for a "which year" question, ranking
      helper columns) -- app/sql/improve.py. The rewrite is used only if it
      still validates;
  2. C5: the machine-checkable business rules for the chosen tables pass
     (e.g. "how many customers" over orders must use COUNT(DISTINCT ...)).
     A violation is an ordinary error, so the retry loop fixes it.
Valid candidates go on to execution. If none is valid, the graph retries
with the first candidate's error.

Refused outright, never retried:
  - any candidate tries to write (DELETE/DROP/UPDATE ...)  -> blocked_reason "write"
  - no candidate is valid and one read hidden data (S3)    -> blocked_reason "restricted"
"""

from app.state.agent_state import AgentState
from app.database.connection import readonly_connection
from app.sql.validator import validate_sql
from app.schema.retriever import check_rules
from app.schema.catalog import readable_columns
from app.sql.repair import repair_alias
from app.sql.improve import improve_sql

REFUSALS = {
    "write": ("Refused: this agent is read-only. The request would modify the "
              "database, so no query was run."),
    "restricted": ("Refused: that question needs data this agent is not allowed "
                   "to read, so no query was run."),
}


def validation_node(state: AgentState) -> dict:
    question = state.get("resolved_input") or state["user_input"]
    tables = state.get("relevant_tables") or []
    checked, history = [], []

    with readonly_connection() as conn:
        for c in state.get("candidates") or [{"sql": state.get("sql_query") or "", "interpretation": None}]:
            conn.last_denial = None
            ok, error = validate_sql(conn, c["sql"])
            for _ in range(3):                      # up to 3 bad references per query
                if ok:
                    break
                fixed = repair_alias(c["sql"], error, readable_columns())
                if not fixed:
                    break
                history.append({"attempt": state.get("attempts", 0), "stage": "validation",
                                "sql": c["sql"], "error": f"{error} -> repaired alias in code"})
                c = {**c, "sql": fixed, "repaired": True}
                conn.last_denial = None
                ok, error = validate_sql(conn, c["sql"])
            denial = conn.last_denial if not ok else None
            if ok:
                improved, notes = improve_sql(c["sql"], question)
                if notes:
                    conn.last_denial = None
                    ok2, err2 = validate_sql(conn, improved)
                    if ok2:
                        history.append({"attempt": state.get("attempts", 0), "stage": "validation",
                                        "sql": c["sql"], "error": "logic fix in code: " + "; ".join(notes)})
                        c = {**c, "sql": improved, "fixes": notes}
            if ok:
                rule_error = check_rules(question, c["sql"], tables)
                if rule_error:
                    ok, error = False, rule_error
            checked.append({**c, "valid": ok, "error": error, "denial": denial})
            history.append({"attempt": state.get("attempts", 0), "stage": "validation",
                            "sql": c["sql"], "error": error})

    valid = [c for c in checked if c["valid"]]
    reason = None
    if any(c["denial"] == "write" for c in checked):
        reason = "write"            # conservative: a write attempt is never ignored
    elif not valid and any(c["denial"] == "restricted" for c in checked):
        reason = "restricted"

    first = valid[0] if valid else checked[0]
    out = {
        "candidates": checked,
        "sql_query": first["sql"],
        "is_valid": bool(valid) and not reason,
        "validation_error": None if valid else first["error"],
        "blocked": bool(reason),
        "blocked_reason": reason,
        "attempt_history": history,
    }
    if reason:
        out["final_answer"] = REFUSALS[reason]
    return out
