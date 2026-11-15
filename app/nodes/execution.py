"""
Node 5: execution
Runs every VALID candidate (fresh read-only connection, timeout + row cap:
S1/S2/S6) and compares their RESULTS. This is the semantic check (C1):
before, any query that ran without an error was accepted, even if it
answered a different question.

- Results that match go in one group. "Match" is lenient: same number of
  rows, and every column of the narrower result appears in the wider one
  (so `SELECT name` and `SELECT name, city` agree).
- The group with the most votes wins (a query produced by two calls counts
  twice); confidence = its share of the votes of the candidates that ran.
- C9: the user is asked only for REAL ambiguity: no majority AND at least two
  of the disagreeing results come from a reply where the model itself
  offered several readings. Disagreement caused only by sampling noise is
  answered with the deterministic (temperature 0) candidate, and the answer
  carries a low-confidence note instead (BIRD q1479: the guess was right,
  but the agent asked).
- C1: if the winning result is EMPTY, the graph regenerates once with a hint
  to check filters/values (empty_result_retry). A second empty result is
  accepted as the real answer.
"""

from app.state.agent_state import AgentState
from app.database.connection import readonly_connection
from app.sql.executor import execute_sql
from app.config import CLARIFY_ON_DISAGREEMENT


def _norm(v):
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 6)
    return v


def _col_vectors(rows):
    if not rows:
        return []
    return [sorted(repr(_norm(r[i])) for r in rows) for i in range(len(rows[0]))]


def same_answer(a_rows: list, b_rows: list) -> bool:
    if len(a_rows) != len(b_rows):
        return False
    if not a_rows:
        return True
    a, b = _col_vectors(a_rows), _col_vectors(b_rows)
    narrow, pool = (a, list(b)) if len(a) <= len(b) else (b, list(a))
    for col in narrow:
        if col not in pool:
            return False
        pool.remove(col)
    return True


def preview(rows: list, columns: list) -> str:
    if not rows:
        return "no rows"
    if len(rows) == 1 and len(columns) == 1:
        return str(rows[0][0])
    first = ", ".join(f"{c}={v}" for c, v in zip(columns, rows[0]))
    return f"{len(rows)} row(s), e.g. {first}"


def execution_node(state: AgentState) -> dict:
    candidates = [dict(c) for c in state.get("candidates") or []]
    valid = [c for c in candidates if c.get("valid")]
    history = []

    with readonly_connection() as conn:
        for c in valid:
            rows, cols, err, trunc = execute_sql(conn, c["sql"])
            c.update(rows=rows, columns=cols, execution_error=err, truncated=trunc)
            if err:
                history.append({"attempt": state.get("attempts", 0), "stage": "execution",
                                "sql": c["sql"], "error": err})

    ok = [c for c in valid if not c.get("execution_error")]
    if not ok:
        first = valid[0] if valid else {"sql": state.get("sql_query"), "execution_error": "no valid query"}
        return {"candidates": candidates, "sql_query": first["sql"], "rows": [], "columns": [],
                "execution_error": first.get("execution_error"), "attempt_history": history}

    groups: list[list[dict]] = []
    for c in ok:
        for g in groups:
            if same_answer(g[0]["rows"], c["rows"]):
                g.append(c)
                break
        else:
            groups.append([c])
    votes = lambda g: sum(c.get("votes", 1) for c in g)
    groups.sort(key=votes, reverse=True)         # stable: ties keep candidate order
    top = groups[0]
    chosen = top[0]
    confidence = round(votes(top) / sum(c.get("votes", 1) for c in ok), 2)
    readings = sum(1 for g in groups if any(c.get("multi") for c in g))
    ambiguous = CLARIFY_ON_DISAGREEMENT and len(groups) > 1 and confidence <= 0.5 and readings >= 2
    if not ambiguous and confidence <= 0.5:
        # no majority, no real ambiguity: trust the deterministic call (call 0)
        top = min(groups, key=lambda g: min(c.get("call", 0) for c in g))
        chosen = min(top, key=lambda c: c.get("call", 0))
        confidence = round(votes(top) / sum(c.get("votes", 1) for c in ok), 2)

    out = {
        "candidates": candidates,
        "sql_query": chosen["sql"],
        "rows": chosen["rows"],
        "columns": chosen["columns"],
        "truncated": chosen.get("truncated"),
        "execution_error": None,
        "confidence": confidence,
        "candidate_groups": len(groups),
        "needs_clarification": ambiguous,
        "attempt_history": history,
    }
    if ambiguous:
        out["clarification_options"] = [{
            "interpretation": next((c["interpretation"] for c in g if c.get("interpretation")),
                                   f"Reading {i + 1}"),
            "sql": g[0]["sql"],
            "preview": preview(g[0]["rows"], g[0]["columns"]),
        } for i, g in enumerate(groups[:3])]
    elif not chosen["rows"] and not state.get("empty_rechecked"):
        out["empty_result_retry"] = True
        out["empty_rechecked"] = True
    return out
