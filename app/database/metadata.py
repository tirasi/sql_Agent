"""
Introspects the live DB -- used as a sanity cross-check against
schema_data/tables.json (e.g. to catch schema drift), not as the primary
source of schema descriptions. The JSON files stay the primary source
because they carry human-written descriptions an LLM prompt benefits from,
which raw introspection doesn't provide.
"""

from app.database.connection import get_connection


def list_tables() -> list[str]:
    conn = get_connection()
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]


def list_columns(table_name: str) -> list[str]:
    conn = get_connection()
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cur.fetchall()]
