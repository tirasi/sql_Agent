"""
Validates SQL without running it for real, using SQLite's own parser via
EXPLAIN. Must be given the READ-ONLY connection: its authorizer rejects
writes at prepare time, so `EXPLAIN DELETE ...` fails here with
"not authorized" instead of passing as valid syntax.

conn.execute() also refuses multiple statements ("SELECT 1; DROP TABLE x").
Depending on the Python version that raises sqlite3.ProgrammingError (3.12+)
or sqlite3.Warning (older), and Warning is NOT a subclass of sqlite3.Error --
so both are caught explicitly.
"""

import sqlite3


def is_write_attempt(error: str | None) -> bool:
    return bool(error) and "not authorized" in error.lower()


def validate_sql(conn: sqlite3.Connection, query: str) -> tuple[bool, str | None]:
    if not query or not query.strip():
        return False, "empty query"
    try:
        conn.execute(f"EXPLAIN {query}")
        return True, None
    except (sqlite3.Error, sqlite3.Warning) as e:
        return False, str(e)
