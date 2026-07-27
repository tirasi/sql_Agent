"""
Validates SQL without mutating/running it for real, using SQLite's own
parser via EXPLAIN. Kept separate from executor.py so validation and real
execution can have different error-handling policies if needed later
(e.g. validation retries silently, execution surfaces to the user).
"""

import sqlite3


def validate_sql(conn: sqlite3.Connection, query: str) -> tuple[bool, str | None]:
    try:
        conn.execute(f"EXPLAIN {query}")
        return True, None
    except sqlite3.Error as e:
        return False, str(e)
