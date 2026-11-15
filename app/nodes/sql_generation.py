"""
Node 3: sql_generation
Writes CANDIDATE queries (C1). The candidates are validated, executed and
compared by result in the next nodes; the majority answer wins, and a
disagreement with no majority becomes a clarifying question (C9).

- Round 1: SQL_CANDIDATES calls -- call 1 at temperature 0, the rest sampled
  at 0.7. Each call may itself return up to 3 queries when it sees several
  readings of the question, each tagged "-- interpretation: ...".
- Retry rounds (after an error, or after an empty result): one call with
  the previous query + the problem.
- If the user just picked a reading from a clarifying question, its SQL is
  reused directly -- no LLM call.

Prompt context added here:
- C7: today's date and the stored date format, so "last month" works.
- C4: exact SQL for business terms ("sales" = SUM(orders.amount)).
- C8: on a follow-up, the previous question and its SQL.
"""

import os
import re
from datetime import date
from app.state.agent_state import AgentState
from app.llm.client import call_llm
from app.config import SQL_CANDIDATES, REFERENCE_DATE

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
MAX_CANDIDATES = 5
EMPTY_RESULT_HINT = (
    "The query ran but returned 0 rows. Check that filter values match the stored "
    "values exactly (see VALUES MENTIONED IN THE QUESTION), that joins use the join "
    "paths, and that no filter was added that the question did not ask for. If 0 rows "
    "is genuinely the correct answer, return the same query again."
)


def _load_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), "r") as f:
        return f.read()


# ---------- C3: robust SQL extraction ----------

_FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_INTERP = re.compile(r"--\s*interpretation\s*:\s*(.+)", re.IGNORECASE)
# a real CTE: WITH [RECURSIVE] name [(cols)] AS (
_CTE = re.compile(r"\bwith\s+(?:recursive\s+)?\w+\s*(?:\([^)]*\))?\s+as\s*\(", re.IGNORECASE)
_LINE_START = re.compile(r"(?im)^[ \t]*(?:select\b|with\s+(?:recursive\s+)?\w+\s*(?:\([^)]*\))?\s+as\s*\()")
_SQL_WORDS = re.compile(r"^\s*(select|from|where|join|left|right|inner|outer|cross|on|and|or|group|order|"
                        r"having|limit|offset|union|except|intersect|with|as|case|when|then|else|end|"
                        r"\)|\(|,|--)", re.IGNORECASE)


def _trim_trailing_prose(sql: str) -> str:
    """Cut explanations the model wrote after the query."""
    if ";" in sql:
        sql = sql[:sql.index(";")]
    kept = []
    for line in sql.splitlines():
        if kept and line.strip() and not _SQL_WORDS.match(line) and re.match(r"^\s*[A-Z][a-z]+\s", line) \
                and not line.startswith((" ", "\t")):
            break        # "This query counts ..." -> stop
        kept.append(line)
    return "\n".join(kept)


def clean_sql(raw: str) -> str:
    """Extract the SQL from a model reply.

    Old version: kept everything from the first word "with" or "select"
    ANYWHERE, so "Here is the query with a join: SELECT ..." became
    "with a join: SELECT ...", which failed and wasted a retry.
    New order: 1) the first ```sql fenced block; 2) a SELECT / real CTE at the
    start of a line; 3) a real CTE anywhere; 4) the first upper-case SELECT,
    else the last "select". Comment lines are dropped and trailing prose is
    cut. If there is no SELECT at all (e.g. the model wrote DELETE), the text
    is kept so the read-only validator can see and block it."""
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("--")).strip()

    start = None
    m = _LINE_START.search(text) or _CTE.search(text)
    if m:
        start = m.start()
    else:
        upper = re.search(r"\bSELECT\b", text)
        if upper:
            start = upper.start()
        else:
            allsel = list(re.finditer(r"\bselect\b", text, re.IGNORECASE))
            if allsel:
                start = allsel[-1].start()
    if start is not None:
        text = _trim_trailing_prose(text[start:])
    return text.strip().rstrip(";").strip()


def parse_candidates(raw: str) -> list[dict]:
    """[{"sql", "interpretation"}] from one reply (one or several fenced queries)."""
    out = []
    fences = list(_FENCE.finditer(raw))
    if fences:
        prev_end = 0
        for f in fences:
            before = raw[prev_end:f.start()]
            interp = _INTERP.findall(before) or _INTERP.findall(f.group(1))
            out.append({"sql": clean_sql(f.group(0)), "interpretation": interp[-1].strip() if interp else None})
            prev_end = f.end()
    else:
        chunks = re.split(r"(?=--\s*interpretation\s*:)", raw, flags=re.IGNORECASE)
        for ch in chunks:
            if not ch.strip():
                continue
            interp = _INTERP.findall(ch)
            sql = clean_sql(ch)
            if sql:
                out.append({"sql": sql, "interpretation": interp[0].strip() if interp else None})
    return [c for c in out if c["sql"]]


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


# ---------- prompt context ----------

def today() -> str:
    return REFERENCE_DATE or date.today().isoformat()


def _date_context() -> str:
    d = today()
    return (f"Today's date is {d}. Dates are stored as TEXT in YYYY-MM-DD format. For relative "
            f"dates use this date, e.g. last 30 days: order_date >= date('{d}', '-30 days'); "
            f"last calendar month: strftime('%Y-%m', order_date) = strftime('%Y-%m', date('{d}', 'start of month', '-1 month')). "
            f"Do not use date('now').")


def _metric_context(state: AgentState) -> str:
    hints = state.get("metric_hints") or []
    if not hints:
        return ""
    return "\nDEFINITIONS (use exactly):\n" + "\n".join(
        f"  - \"{h['term']}\" means {h['sql']}  ({h['description']})" for h in hints) + "\n"


def _followup_context(state: AgentState) -> str:
    history = state.get("history") or []
    if not (state.get("is_follow_up") and history and history[-1].get("sql")):
        return ""
    last = history[-1]
    return ("\nThe new question is a FOLLOW-UP to this previous question. Change the previous "
            "query to answer the new one.\n"
            f"  Previous question: {last['question']}\n"
            f"  Previous SQL: {' '.join(last['sql'].split())}\n")


def sql_generation_node(state: AgentState) -> dict:
    attempts = state.get("attempts", 0)

    if state.get("chosen_sql") and attempts == 0:      # C9: user picked a reading
        cands = [{"sql": state["chosen_sql"], "interpretation": state.get("chosen_interpretation"), "votes": 1}]
        return {"candidates": cands, "sql_query": cands[0]["sql"], "attempts": 1,
                "execution_error": None, "empty_result_retry": False}

    question = state.get("resolved_input") or state["user_input"]
    ctx = dict(
        schema_context=state.get("schema_context") or "",
        join_context=state.get("join_context") or "(single table -- no joins needed)",
        date_context=_date_context(),
        metric_context=_metric_context(state),
        followup_context=_followup_context(state),
    )
    if state.get("empty_result_retry"):
        problem = EMPTY_RESULT_HINT
    else:
        problem = state.get("validation_error") or state.get("execution_error")

    if attempts == 0 or not problem:
        system_prompt = _load_prompt("sql_generation.md").format(**ctx)
        temps = [0.0] + [0.7] * max(0, SQL_CANDIDATES - 1)
    else:
        system_prompt = _load_prompt("sql_retry.md").format(
            error=problem, previous_query=state.get("sql_query"), **ctx)
        temps = [0.0]

    # identical SQL from two calls is run once but counts as two votes:
    # the model producing the same query again is evidence of agreement
    cands, by_key = [], {}
    for t in temps:
        for c in parse_candidates(call_llm(system_prompt, question, node_name="sql_generation", temperature=t)):
            key = _norm_sql(c["sql"])
            if key in by_key:
                by_key[key]["votes"] += 1
                by_key[key]["interpretation"] = by_key[key]["interpretation"] or c["interpretation"]
            elif len(cands) < MAX_CANDIDATES:
                by_key[key] = {**c, "votes": 1}
                cands.append(by_key[key])
    cands = cands or [{"sql": "", "interpretation": None, "votes": 1}]

    return {
        "candidates": cands,
        "sql_query": cands[0]["sql"],
        "attempts": attempts + 1,
        "execution_error": None,      # clear stale errors from the previous round
        "empty_result_retry": False,
    }
