"""
Ask questions in plain English, get the SQL (and the result) back.

    python app/ask.py                                  # interactive, on DB_PATH from .env (dummy.db)
    python app/ask.py "How many customers are there?"  # one question, then exit
    python app/ask.py --db ~/bird/.../california_schools/california_schools.sqlite
    python app/ask.py --sql-only                       # print only the SQL (no result table / answer)

Interactive commands:  /tables  /reset (forget the conversation)  /help  exit
Follow-ups work ("what about Mumbai?"), and if the agent asks which reading
you meant, answer with the number.

With --db, any SQLite file works: the schema is read from the file itself
(plus BIRD-style database_description/*.csv if present), every table is
visible, and nothing in the file can be changed (read-only connection).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _setup_env(args) -> None:
    """Must run BEFORE any app.* import: config.py reads env vars at import."""
    if args.db:
        db = Path(args.db).expanduser().resolve()
        if not db.exists():
            sys.exit(f"No such database file: {db}")
        from app.schema.bootstrap import build_schema_dir
        schema_dir = Path(tempfile.mkdtemp(prefix="nlq_ask_")) / "schema"
        build_schema_dir(db, schema_dir)
        os.environ.update(DB_PATH=str(db), SCHEMA_DATA_DIR=str(schema_dir), AUTO_EXPOSE_SCHEMA="true")
    if args.candidates:
        os.environ["SQL_CANDIDATES"] = str(args.candidates)


def pretty_sql(sql: str | None) -> str:
    if not sql:
        return "(no SQL)"
    try:
        import sqlglot
        return sqlglot.transpile(sql, read="sqlite", write="sqlite", pretty=True)[0]
    except Exception:
        return sql


def table(columns: list, rows: list, limit: int) -> str:
    if not columns:
        return "(no columns)"
    shown = [[("NULL" if v is None else str(v)) for v in r] for r in rows[:limit]]
    widths = [min(40, max([len(str(c))] + [len(r[i]) for r in shown])) for i, c in enumerate(columns)]
    fmt = lambda vals: " | ".join(str(v)[:w].ljust(w) for v, w in zip(vals, widths))
    lines = [fmt(columns), "-+-".join("-" * w for w in widths)] + [fmt(r) for r in shown]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more row(s)")
    return "\n".join(lines)


def format_state(state: dict, sql_only: bool = False, limit: int = 10) -> str:
    """What the user sees for one question."""
    out = []
    if state.get("blocked"):
        return state.get("final_answer") or "Refused."
    if state.get("needs_clarification"):
        out.append(state["final_answer"])
        return "\n".join(out)
    if state.get("execution_error") or not state.get("is_valid"):
        return state.get("final_answer") or "Could not produce a working query."
    out += ["SQL:", pretty_sql(state.get("sql_query"))]
    fixes = [h["error"] for h in state.get("attempt_history") or []
             if (h.get("error") or "").startswith("logic fix") or "repaired alias" in (h.get("error") or "")]
    for f in fixes:
        out.append(f"  (fixed in code: {f.replace('logic fix in code: ', '')})")
    if sql_only:
        return "\n".join(out)
    rows, cols = state.get("rows") or [], state.get("columns") or []
    out += ["", f"Result ({len(rows)} row(s)):", table(cols, rows, limit)]
    if state.get("final_answer"):
        out += ["", "Answer: " + state["final_answer"]]
    conf, n = state.get("confidence"), len(state.get("candidates") or [])
    if conf is not None and n > 1:
        out.append(f"Confidence: {round(conf * 100)}% of {n} candidate queries agreed")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask a question, get the SQL.")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--db", help="any SQLite file (default: DB_PATH from .env)")
    ap.add_argument("--candidates", type=int, help="SQL candidates per question (1 = fastest)")
    ap.add_argument("--sql-only", action="store_true", help="show only the SQL")
    ap.add_argument("--rows", type=int, default=10, help="rows to show (default 10)")
    args = ap.parse_args(argv)
    words = " ".join(args.question)
    if args.question and (".py" in words or words.startswith(("python", "./", "cd "))):
        sys.exit(f'"{words}" looks like a command, not a question -- two commands were probably '
                 "pasted on one line. Run them one at a time.")
    _setup_env(args)

    from app.graph.nlq_sql_graph import build_graph
    from app.state.agent_state import make_initial_state, make_turn
    from app.llm.client import check_provider_ready
    from app.schema.catalog import get_catalog
    from app.config import DB_PATH, HISTORY_TURNS

    problems = check_provider_ready()
    if problems:
        sys.exit("Not ready:\n  - " + "\n  - ".join(problems))
    graph = build_graph()
    history: list = []

    def ask(q: str) -> None:
        state = graph.invoke(make_initial_state(q, history[-HISTORY_TURNS:]))
        history.append(make_turn(state))
        print(format_state(state, args.sql_only, args.rows))

    if args.question:
        ask(" ".join(args.question))
        return

    try:
        import readline  # noqa: F401  (arrow keys / history in the prompt)
    except ImportError:
        pass
    print(f"Database: {DB_PATH}   ({len(get_catalog())} tables)   type /help for commands")
    while True:
        try:
            q = input("\nAsk> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q", "/exit"):
            break
        if q == "/help":
            print("Type a question in plain English. Follow-ups work. Commands: /tables  /reset  exit")
            continue
        if q == "/tables":
            for t, m in get_catalog().items():
                print(f"  {t}: {', '.join(m['columns'])}")
            continue
        if q == "/reset":
            history.clear()
            print("Conversation forgotten.")
            continue
        try:
            ask(q)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
