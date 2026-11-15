"""
Benchmark the agent on a REAL text-to-SQL dataset (BIRD mini-dev, BIRD dev or
Spider dev) and report execution accuracy (EX).

EX (the official BIRD metric): a question is correct when the SET of rows the
agent's SQL returns equals the set of rows the gold SQL returns.

Scores reported:
  - EX strict  : official metric; a clarifying question counts as WRONG
                 (this is the number comparable to leaderboards)
  - EX guess   : the agent's best guess (top-voted candidate) is scored anyway
  - EX lenient : NOT official. Right rows, but extra columns allowed (every
                 gold column must appear in the result). The gap between
                 lenient and strict = answers whose logic was right but that
                 returned helper columns the question didn't ask for.

Usage (from the project root, Ollama running or LLM_PROVIDER=anthropic):
    python tests/run_benchmark.py \
        --questions ~/bird/minidev/MINIDEV/mini_dev_sqlite.json \
        --db-root   ~/bird/minidev/MINIDEV/dev_databases \
        --limit 50 --candidates 1

    --db california_schools   only one database
    --sample 100              random 100 questions (seeded), instead of the first N
    --no-evidence             don't give BIRD's per-question hint ("evidence")
    --no-clarify              never ask; always answer with the best guess
    --oracle                  feed the GOLD SQL as the model's answer: checks the
                              harness end to end without any LLM (should be ~100%)
    --resume FILE.jsonl       continue an interrupted run

Each database runs in its own subprocess, because the app reads DB_PATH and the
schema folder once at import time. Results are appended to a .jsonl file after
every question, so a crash or Ctrl-C loses nothing.
"""

import argparse
import csv
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent


# ---------------------------------------------------------------- parent side

def find_db_file(db_root: Path, db_id: str) -> Path:
    for name in (f"{db_id}.sqlite", f"{db_id}.db"):
        p = db_root / db_id / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no {db_id}.sqlite under {db_root / db_id}")


def _read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                        for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    return []


def build_schema_dir(db_file: Path, out_dir: Path) -> None:
    """tables.json for one benchmark DB: every real table/column, with BIRD's
    database_description/*.csv text as descriptions when present."""
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    tables = {}
    desc_dir = db_file.parent / "database_description"
    csv_by_table = {p.stem.lower(): p for p in desc_dir.glob("*.csv")} if desc_dir.exists() else {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        cols = {r[1]: (r[2] or "untyped") for r in conn.execute(f'PRAGMA table_info("{t}")')}
        by_lower = {c.lower(): c for c in cols}
        if t.lower() in csv_by_table:
            for row in _read_csv(csv_by_table[t.lower()]):
                real = by_lower.get((row.get("original_column_name") or "").lower())
                if not real:
                    continue
                parts = [cols[real]]
                for key in ("column_description", "value_description"):
                    v = " ".join((row.get(key) or "").split())
                    if v and v.lower() not in ("nan", "none"):
                        parts.append(v[:300])
                cols[real] = " - ".join(parts)
        tables[t] = {"description": f"Table {t}.", "columns": cols}
    conn.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=1))
    for name, empty in (("relationships.json", []), ("business_rules.json", []),
                        ("synonyms.json", {"values": {}, "tables": {}}), ("metrics.json", {}),
                        ("access_policy.json", {"hidden_columns": {}})):
        (out_dir / name).write_text(json.dumps(empty))


def load_cases(path: Path) -> list[dict]:
    raw = json.loads(path.expanduser().read_text())
    cases = []
    for i, c in enumerate(raw):
        cases.append({
            "id": c.get("question_id", i),
            "db_id": c["db_id"],
            "question": c["question"],
            "evidence": c.get("evidence") or "",
            "gold_sql": c.get("SQL") or c.get("query"),       # BIRD: SQL, Spider: query
            "difficulty": c.get("difficulty", "n/a"),
        })
    return cases


def summarise(results: list[dict], out_path: Path) -> None:
    scored = [r for r in results if not r.get("gold_error")]
    n = len(scored)
    if not n:
        print("No scored questions.")
        return

    def pct(a, b):
        return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "  n/a"

    strict = sum(r["ex_strict"] for r in scored)
    guess = sum(r["ex_guess"] for r in scored)
    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Questions scored     : {n}  (skipped, gold SQL failed: {len(results) - n})")
    print(f"EX strict            : {pct(strict, n)}   <- clarifying question = wrong")
    print(f"EX best guess        : {pct(guess, n)}")
    lenient = sum(r.get("ex_lenient", r["ex_guess"]) for r in scored)
    print(f"EX lenient (unofficial, extra columns OK): {pct(lenient, n)}")
    print(f"Asked to clarify     : {pct(sum(r['clarified'] for r in scored), n)}")
    print(f"Refused / blocked    : {pct(sum(bool(r['blocked_reason']) for r in scored), n)}")
    print(f"No SQL ran           : {pct(sum(r['pred_failed'] for r in scored), n)}")
    print(f"Avg latency          : {sum(r['latency_s'] for r in scored) / n:.1f}s per question")

    print("\nBy difficulty:")
    groups = defaultdict(list)
    for r in scored:
        groups[r["difficulty"]].append(r)
    for k in sorted(groups):
        print(f"  {k:<12} {pct(sum(r['ex_strict'] for r in groups[k]), len(groups[k]))}")

    print("\nBy database:")
    groups = defaultdict(list)
    for r in scored:
        groups[r["db_id"]].append(r)
    for k in sorted(groups):
        print(f"  {k:<26} {pct(sum(r['ex_strict'] for r in groups[k]), len(groups[k]))}")

    # Does the confidence score mean anything? Accuracy when all candidates agreed vs not.
    agreed = [r for r in scored if r["confidence"] == 1 and not r["clarified"]]
    split = [r for r in scored if r["confidence"] is not None and r["confidence"] < 1 and not r["clarified"]]
    if agreed and split:
        print("\nConfidence check (is the agent right more often when its candidates agree?):")
        print(f"  all candidates agreed : {pct(sum(r['ex_guess'] for r in agreed), len(agreed))}")
        print(f"  candidates disagreed  : {pct(sum(r['ex_guess'] for r in split), len(split))}")
    print(f"\nPer-question details: {out_path}")


def parent(args) -> None:
    cases = load_cases(Path(args.questions))
    if args.db:
        cases = [c for c in cases if c["db_id"] == args.db]
    if args.sample:
        random.Random(42).shuffle(cases)
        cases = cases[:args.sample]
    elif args.limit:
        cases = cases[:args.limit]

    out_path = Path(args.resume) if args.resume else \
        TESTS / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} questions selected, {len(done)} already done, {len(todo)} to run -> {out_path}")

    db_root = Path(args.db_root).expanduser()
    by_db = defaultdict(list)
    for c in todo:
        by_db[c["db_id"]].append(c)

    work = Path(tempfile.mkdtemp(prefix="nlq_bench_"))
    for db_id, db_cases in by_db.items():
        db_file = find_db_file(db_root, db_id)
        schema_dir = work / db_id / "schema"
        build_schema_dir(db_file, schema_dir)
        cases_file = work / db_id / "cases.json"
        cases_file.write_text(json.dumps(db_cases))
        env = {**os.environ,
               "DB_PATH": str(db_file),
               "SCHEMA_DATA_DIR": str(schema_dir),
               "AUTO_EXPOSE_SCHEMA": "true",          # benchmark DBs: expose every table
               "QUERY_TIMEOUT_MS": str(args.timeout_ms),
               "MAX_RESULT_ROWS": "100000",            # EX compares full results
               "SQL_CANDIDATES": str(args.candidates),
               "CLARIFY_ON_DISAGREEMENT": "false" if args.no_clarify else "true"}
        print(f"\n=== {db_id}: {len(db_cases)} question(s) ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--cases", str(cases_file), "--out", str(out_path)]
        if args.no_evidence:
            cmd.append("--no-evidence")
        if args.oracle:
            cmd.append("--oracle")
        rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
        if rc != 0:
            print(f"!! worker for {db_id} exited with code {rc}; re-run with --resume {out_path} to continue")
            if rc == 2:   # provider not ready -> pointless to continue
                break

    results = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()] if out_path.exists() else []
    wanted = {c["id"] for c in cases}
    summarise([r for r in results if r["id"] in wanted], out_path)


# ---------------------------------------------------------------- worker side

def lenient_match(pred: list, gold: list) -> bool:
    """Same distinct rows, ignoring extra predicted columns."""
    pred, gold = list(set(map(tuple, pred))), list(set(map(tuple, gold)))
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    cols = lambda rows: [sorted(map(repr, (r[i] for r in rows))) for i in range(len(rows[0]))]
    pool = cols(pred)
    for col in cols(gold):
        if col not in pool:
            return False
        pool.remove(col)
    return True


def worker(args) -> None:
    sys.path.insert(0, str(ROOT))
    from unittest import mock
    from app.graph.nlq_sql_graph import build_graph
    from app.state.agent_state import make_initial_state
    from app.llm.client import check_provider_ready
    import app.nodes.response_generation as rg
    import app.nodes.sql_generation as sg
    import app.nodes.schema_retrieval as sr

    if not args.oracle:
        problems = check_provider_ready()
        if problems:
            print("Not ready:\n  - " + "\n  - ".join(problems))
            sys.exit(2)

    cases = json.loads(Path(args.cases).read_text())
    graph = build_graph()
    gold_conn = sqlite3.connect(f"file:{os.environ['DB_PATH']}?mode=ro", uri=True)

    def run_gold(sql):
        deadline = time.monotonic() + 60
        gold_conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            return gold_conn.execute(sql).fetchall(), None
        except sqlite3.Error as e:
            return None, str(e)
        finally:
            gold_conn.set_progress_handler(None, 0)

    patches = [mock.patch.object(rg, "call_llm", return_value="")]   # EX needs SQL only: skip the answer LLM
    current = {}
    if args.oracle:
        patches += [
            mock.patch.object(sg, "call_llm", side_effect=lambda *a, **k: f"```sql\n{current['gold']}\n```"),
            mock.patch.object(sr, "call_llm", return_value="[]"),
        ]
    for p in patches:
        p.start()

    with open(args.out, "a") as out:
        for c in cases:
            current["gold"] = c["gold_sql"]
            q = c["question"]
            if c["evidence"] and not args.no_evidence:
                q = f"{q}\nHint: {c['evidence']}"
            gold_rows, gold_error = run_gold(c["gold_sql"])
            t0 = time.perf_counter()
            crash = None
            try:
                state = graph.invoke(make_initial_state(q))
            except Exception as e:                       # record, don't stop the run
                state, crash = {}, f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0

            ran = state.get("execution_error") is None and state.get("rows") is not None and not state.get("blocked") \
                and not crash and state.get("is_valid")
            pred = state.get("rows") or []
            ex_guess = bool(ran and gold_rows is not None and set(map(tuple, pred)) == set(map(tuple, gold_rows)))
            clarified = bool(state.get("needs_clarification"))
            ex_lenient = bool(ran and gold_rows is not None and (ex_guess or lenient_match(pred, gold_rows)))
            r = {
                "id": c["id"], "db_id": c["db_id"], "difficulty": c["difficulty"],
                "question": c["question"], "evidence": c["evidence"],
                "gold_sql": c["gold_sql"], "pred_sql": state.get("sql_query"),
                "ex_guess": ex_guess, "ex_strict": ex_guess and not clarified, "ex_lenient": ex_lenient,
                "clarified": clarified,
                "blocked_reason": state.get("blocked_reason"), "pred_failed": not ran,
                "confidence": state.get("confidence"),
                "n_candidates": len(state.get("candidates") or []), "attempts": state.get("attempts"),
                "pred_error": crash or state.get("execution_error") or state.get("validation_error"),
                "gold_error": gold_error, "latency_s": round(latency, 2),
                "relevant_tables": state.get("relevant_tables"),
                "candidates": [{k: c.get(k) for k in ("sql", "interpretation", "votes", "valid", "error", "fixes",
                                                      "execution_error", "multi")}
                               for c in state.get("candidates") or []],
                "trace": state.get("attempt_history"),
            }
            out.write(json.dumps(r, default=str) + "\n")
            out.flush()
            mark = "SKIP" if gold_error else ("PASS" if r["ex_strict"] else ("ASK " if clarified else "FAIL"))
            print(f"[{mark}] {c['id']} ({c['difficulty']}, {latency:.1f}s) {c['question'][:90]}", flush=True)
    for p in patches:
        p.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", help="BIRD/Spider question json")
    ap.add_argument("--db-root", help="folder with <db_id>/<db_id>.sqlite")
    ap.add_argument("--db", help="only this db_id")
    ap.add_argument("--limit", type=int, help="first N questions")
    ap.add_argument("--sample", type=int, help="random N questions (seed 42)")
    ap.add_argument("--candidates", type=int, default=int(os.getenv("SQL_CANDIDATES", "3")))
    ap.add_argument("--timeout-ms", type=int, default=30000, help="per-query timeout for the agent's SQL")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--no-clarify", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--resume", help="jsonl file of an earlier run to continue")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cases", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        if not (args.questions and args.db_root):
            ap.error("--questions and --db-root are required")
        parent(args)


if __name__ == "__main__":
    main()
