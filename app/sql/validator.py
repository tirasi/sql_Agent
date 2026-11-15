"""
Validates SQL without running it for real, using SQLite's own parser via
EXPLAIN. Must be given the READ-ONLY connection: its authorizer rejects
writes, unlisted functions and uncatalogued columns at prepare time, so
`EXPLAIN DELETE ...` fails here with "not authorized" instead of passing.

conn.execute() also refuses multiple statements ("SELECT 1; DROP TABLE x").
Depending on the Python version that raises sqlite3.ProgrammingError (3.12+)
or sqlite3.Warning (older), and Warning is NOT a subclass of sqlite3.Error --
so both are caught explicitly.
"""

import sqlite3


# String helpers kept for scripts/tests; the validation node uses
# conn.last_denial (set by the authorizer) instead.
def is_write_attempt(error: str | None) -> bool:
    """Write / PRAGMA / ATTACH denied by the authorizer. A denied FUNCTION
    (S5, 'not authorized to use function: randomblob') is NOT a write: it is
    an ordinary validation error the retry loop can fix."""
    e = (error or "").lower()
    return "not authorized" in e and "use function" not in e


def is_restricted_read(error: str | None) -> bool:
    """S3: read of a table/column outside the catalog or access policy.
    SQLite words a denied SQLITE_READ as 'access to X.Y is prohibited'."""
    return bool(error) and "prohibited" in error.lower()


def validate_sql(conn: sqlite3.Connection, query: str) -> tuple[bool, str | None]:
    if not query or not query.strip():
        return False, "empty query"
    try:
        conn.execute(f"EXPLAIN {query}")
        return True, None
    except (sqlite3.Error, sqlite3.Warning) as e:
        return False, str(e)
