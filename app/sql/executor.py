import sqlite3
import time

from app.config import QUERY_TIMEOUT_MS, MAX_RESULT_ROWS

TIMEOUT_ERROR = "query timed out"


def execute_sql(conn: sqlite3.Connection, query: str,
                timeout_ms: int = QUERY_TIMEOUT_MS,
                max_rows: int = MAX_RESULT_ROWS) -> tuple[list, list, str | None, bool]:
    """Returns (rows, column_names, error, truncated).

    S1: a progress handler aborts the query once it runs longer than
        timeout_ms (endless WITH RECURSIVE, huge cross joins ...).
    S2: at most max_rows rows are fetched; `truncated` says whether more
        existed. fetchmany() stops SQLite early instead of materialising
        everything with fetchall().
    On failure rows/columns are empty and error is populated."""
    deadline = time.monotonic() + timeout_ms / 1000
    timed_out = False

    def _watchdog():
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1          # non-zero = interrupt the running statement
        return 0

    conn.set_progress_handler(_watchdog, 10_000)   # checked every 10k VM steps
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return rows[:max_rows], columns, None, truncated
    except (sqlite3.Error, sqlite3.Warning) as e:
        if timed_out:
            return [], [], f"{TIMEOUT_ERROR} after {timeout_ms} ms -- write a cheaper query (filter early, avoid cross joins and unbounded recursion)", False
        return [], [], str(e), False
    finally:
        conn.set_progress_handler(None, 0)
