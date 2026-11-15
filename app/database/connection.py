"""
Database connections. SQLite for local dev/testing right now.

Two connections on purpose:
- get_connection():          read/write. Only used for seeding the dummy DB
                             and introspection (PRAGMA table_info etc.).
- get_readonly_connection(): what LLM-generated SQL runs on. Three
                             independent layers stop writes (defense in depth):
    1. file opened with mode=ro        -> OS-level: SQLite can't write the file
    2. PRAGMA query_only = ON          -> SQLite refuses any write statement
    3. set_authorizer allow-list       -> only SELECT / column reads / functions
                                          are allowed; INSERT, UPDATE, DELETE,
                                          DROP, ATTACH, PRAGMA... are denied
                                          at PREPARE time (so even EXPLAIN of
                                          a DELETE is rejected).
"""

import os
import sqlite3
from app.config import DB_PATH


_connection: sqlite3.Connection | None = None
_readonly_connection: sqlite3.Connection | None = None

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE ... CTEs
}


def _readonly_authorizer(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def get_connection() -> sqlite3.Connection:
    """Read/write singleton, creating + seeding the dummy DB on first call."""
    global _connection
    if _connection is None:
        is_new = DB_PATH == ":memory:" or not os.path.exists(DB_PATH)
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        if is_new:
            _seed_dummy_data(_connection)
    return _connection


def get_readonly_connection() -> sqlite3.Connection:
    """Connection for running model-written SQL. Cannot modify the DB."""
    global _readonly_connection
    if _readonly_connection is None:
        get_connection()  # make sure the file exists / is seeded first
        if DB_PATH == ":memory:":
            raise RuntimeError("Read-only mode needs a file DB; set DB_PATH to a .db file.")
        uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")      # must run BEFORE the authorizer (it denies PRAGMA)
        conn.set_authorizer(_readonly_authorizer)
        _readonly_connection = conn
    return _readonly_connection


def _seed_dummy_data(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            signup_date TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product TEXT,
            amount REAL,
            order_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)
    cur.executemany(
        "INSERT INTO customers (id, name, city, signup_date) VALUES (?, ?, ?, ?)",
        [
            (1, "Asha Patel", "Jaipur", "2024-01-15"),
            (2, "Ravi Kumar", "Mumbai", "2024-03-02"),
            (3, "Meera Shah", "Jaipur", "2024-05-20"),
            (4, "Kabir Singh", "Delhi", "2024-06-11"),
        ],
    )
    cur.executemany(
        "INSERT INTO orders (id, customer_id, product, amount, order_date) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, "Notebook", 250.0, "2024-02-01"),
            (2, 1, "Pen Set", 90.0, "2024-02-15"),
            (3, 2, "Backpack", 1200.0, "2024-03-10"),
            (4, 3, "Notebook", 250.0, "2024-06-01"),
            (5, 4, "Notebook", 260.0, "2024-06-15"),
            (6, 4, "Backpack", 1150.0, "2024-07-01"),
        ],
    )
    conn.commit()
