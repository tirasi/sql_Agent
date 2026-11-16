"""
Evaluation harness for the NLQ-to-SQL agent.

Quick check on the small dummy.db (23 questions, incl. 3 safety cases).
Runs every question in bench/dummy_eval.json through the full LangGraph
pipeline and reports:
  - execution accuracy  : agent's result set matches the gold query's result
  - first-try validity  : SQL passed validation on attempt 1 (no retries)
  - retry recoveries    : failed on attempt 1, fixed by the retry loop
  - no-answer rate      : pipeline ended without a final answer
  - safety              : destructive requests left the database unchanged
  - average latency per question

Works on a temporary COPY of dummy.db, so your real DB is never modified.

Usage (from the project root, with Ollama running):
    python bench/quick_eval.py
    python bench/quick_eval.py --only 7,8,9      # run selected question ids
"""

import argparse
import hashlib
import json
import re
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent

# Point the app at a throwaway copy of the DB BEFORE importing app.* modules
# (config.py reads DB_PATH at import time; load_dotenv won't override it).
_tmpdir = tempfile.mkdtemp(prefix="nlq_eval_")
EVAL_DB = os.path.join(_tmpdir, "eval.db")
shutil.copy(ROOT / "dummy.db", EVAL_DB)
os.environ["DB_PATH"] = EVAL_DB
os.chdir(ROOT)                      # so relative paths like schema_data/ resolve
sys.path.insert(0, str(ROOT))

from app.graph.nlq_sql_graph import build_graph  # noqa: E402
from app.config import LLM_PROVIDER, NODE_MODELS_OLLAMA  # noqa: E402
from app.state.agent_state import make_initial_state  # noqa: E402
from app.llm.client import check_provider_ready  # noqa: E402


def initial_state(question: str) -> dict:
    return make_initial_state(question)


def gold_tables(sql: str) -> set[str]:
    """Tables the gold SQL uses (FROM/JOIN targets) -- for routing recall."""
    return {t.lower() for t in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.IGNORECASE)}


def _norm(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _columns(rows):
    """Transpose rows into a list of sorted, normalised columns."""
    if not rows:
        return []
    return [sorted((_norm(r[i]) for r in rows), key=repr) for i in range(len(rows[0]))]


def results_match(pred_rows, gold_rows) -> bool:
    """Lenient execution match: same number of rows, and every gold column
    appears among the predicted columns (so SELECT * or extra columns still
    pass, but missing/wrong values fail). Row order is ignored."""
    pred_rows, gold_rows = list(pred_rows or []), list(gold_rows or [])
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    pred_cols = _columns(pred_rows)
    remaining = list(pred_cols)
    for gc in _columns(gold_rows):
        if gc in remaining:
            remaining.remove(gc)
        else:
            return False
    return True


def db_fingerprint(path: str) -> str:
    conn = sqlite3.connect(path)
    h = hashlib.sha256()
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        h.update(name.encode())
        for row in conn.execute(f"SELECT * FROM {name} ORDER BY 1"):
            h.update(repr(row).encode())
    conn.close()
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated question ids to run")
    args = ap.parse_args()

    cases = json.loads((BENCH / "dummy_eval.json").read_text())
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    problems = check_provider_ready()
    if problems:  # fail once, clearly -- not 23 identical crashes
        print("Not ready:\n  - " + "\n  - ".join(problems))
        sys.exit(1)

    graph = build_graph()
    gold_conn = sqlite3.connect(str(ROOT / "dummy.db"))  # read-only use: gold answers
    results = []

    print(f"Provider: {LLM_PROVIDER}  models: {NODE_MODELS_OLLAMA if LLM_PROVIDER == 'ollama' else '(anthropic)'}")
    print(f"Running {len(cases)} questions...\n")

    for case in cases:
        before = db_fingerprint(EVAL_DB)
        t0 = time.perf_counter()
        crash = None
        try:
            state = graph.invoke(initial_state(case["question"]))
        except Exception as e:  # a crash is a result too -- record it, don't stop
            state, crash = initial_state(case["question"]), f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        db_changed = db_fingerprint(EVAL_DB) != before
        if db_changed:  # restore so one bad query can't poison later questions
            shutil.copy(ROOT / "dummy.db", EVAL_DB)

        attempts = state.get("attempts") or 0
        ran_ok = bool(state.get("is_valid")) and not state.get("execution_error") and not crash

        routing_ok = None
        if case["category"] == "safety":
            passed = not db_changed and not crash  # a crash proves nothing about safety
        else:
            gold = gold_conn.execute(case["gold_sql"]).fetchall()
            passed = ran_ok and results_match(state.get("rows"), gold)
            picked = {t.lower() for t in (state.get("relevant_tables") or [])}
            routing_ok = gold_tables(case["gold_sql"]) <= picked

        r = {
            "id": case["id"], "category": case["category"], "question": case["question"],
            "passed": passed, "sql": state.get("sql_query"), "attempts": attempts,
            "valid": state.get("is_valid"), "validation_error": state.get("validation_error"),
            "execution_error": state.get("execution_error"), "crash": crash,
            "rows": state.get("rows"), "answer": state.get("final_answer"),
            "db_changed": db_changed, "latency_s": round(latency, 2),
            "resolved_input": state.get("resolved_input"),
            "relevant_tables": state.get("relevant_tables"),
            "table_selection": state.get("table_selection"),
            "routing_ok": routing_ok, "blocked": state.get("blocked"),
            "confidence": state.get("confidence"),
            "needs_clarification": state.get("needs_clarification"),
            "candidates": [c.get("sql") for c in state.get("candidates") or []],
            "attempt_history": state.get("attempt_history"),
        }
        results.append(r)
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] #{case['id']:>2} ({case['category']}, {attempts} attempt(s), {latency:.1f}s) {case['question']}")
        if not passed:
            print(f"        sql: {r['sql']}")
            for k in ("validation_error", "execution_error", "crash"):
                if r[k]:
                    print(f"        {k}: {r[k]}")
            if r["db_changed"]:
                print("        !! DATABASE WAS MODIFIED")

    # ---- summary ----
    acc_cases = [r for r in results if r["category"] != "safety"]
    safety = [r for r in results if r["category"] == "safety"]
    n = len(acc_cases)

    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    passed = sum(r["passed"] for r in acc_cases)
    first_try = sum(1 for r in acc_cases if r["attempts"] == 1 and r["valid"])
    recovered = sum(1 for r in acc_cases if r["attempts"] > 1 and r["valid"])
    no_answer = sum(1 for r in acc_cases if not r["answer"])

    print("\n========== SUMMARY ==========")
    print(f"Execution accuracy : {pct(passed, n)}")
    print(f"Asked to clarify   : {pct(sum(bool(r.get('needs_clarification')) for r in acc_cases), n)}")
    print(f"Valid on 1st try   : {pct(first_try, n)}")
    print(f"Fixed by retry     : {recovered}")
    print(f"No final answer    : {pct(no_answer, n)}")
    print(f"Routing recall     : {pct(sum(bool(r['routing_ok']) for r in acc_cases), n)}")
    print(f"Safety (DB intact) : {pct(sum(r['passed'] for r in safety), len(safety))}")
    print(f"Writes blocked     : {pct(sum(bool(r['blocked']) for r in safety), len(safety))}")
    print(f"Avg latency        : {sum(r['latency_s'] for r in results) / max(len(results), 1):.1f}s")
    print("By category:")
    cats = sorted({r["category"] for r in acc_cases})
    for c in cats:
        rs = [r for r in acc_cases if r["category"] == c]
        print(f"  {c:<10} {pct(sum(r['passed'] for r in rs), len(rs))}")

    (BENCH / "results").mkdir(exist_ok=True)
    out = BENCH / "results" / f"quick_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
