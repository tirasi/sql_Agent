import sqlite3


def execute_sql(conn: sqlite3.Connection, query: str) -> tuple[list, list, str | None]:
    """Returns (rows, column_names, error). On failure rows/columns are empty
    and error is populated -- callers check `error` rather than catching
    exceptions themselves."""
    try:
        cur = conn.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return rows, columns, None
    except sqlite3.Error as e:
        return [], [], str(e)
