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
    The authorizer also allows reading ONLY catalogued columns (S3) and
    calling ONLY allow-listed functions (S5). A fresh connection is opened
    per request (S6).
"""

import json
import os
import sqlite3
from contextlib import contextmanager
import threading
from functools import lru_cache
from app.config import DB_PATH, SCHEMA_DATA_DIR, ACCESS_POLICY_FILE

#checks if connection exists, if yes then sets it connection, else false
_connection: sqlite3.Connection | None = None
_connection_lock = threading.Lock()   # S6: two requests can't seed the DB at once

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE ... CTEs (S1 timeout caps runaway loops)
}

# S5: only these SQL functions may be called by generated SQL. Anything else
# (randomblob, zeroblob, load_extension, sqlite_* internals ...) is denied.
_ALLOWED_FUNCTIONS = {
    # aggregates
    "count", "sum", "total", "avg", "min", "max", "group_concat", "string_agg",
    # text
    "lower", "upper", "length", "substr", "substring", "trim", "ltrim", "rtrim",
    "replace", "instr", "like", "glob", "printf", "format", "concat", "concat_ws",
    # numbers / null handling
    "abs", "round", "coalesce", "ifnull", "nullif", "iif", "typeof",
    # dates
    "date", "time", "datetime", "julianday", "strftime", "unixepoch",
    # window functions
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead",
    "first_value", "last_value", "nth_value", "percent_rank", "cume_dist",
}


@lru_cache(maxsize=1)
def _readable_columns() -> dict[str, set[str]]:
    """S3: table -> readable columns. Built from schema_data/tables.json (the
    catalog the LLM is shown) minus access_policy.json hidden_columns.
    Anything not in the catalog -- sqlite_master, undocumented tables or
    columns -- is unreadable."""
    with open(os.path.join(SCHEMA_DATA_DIR, "tables.json")) as f:
        tables = json.load(f)
    hidden: dict = {}
    policy_path = os.path.join(SCHEMA_DATA_DIR, ACCESS_POLICY_FILE)
    if os.path.exists(policy_path):
        with open(policy_path) as f:
            hidden = json.load(f).get("hidden_columns", {})
    out = {}
    for table, meta in tables.items():
        blocked = {c.lower() for c in hidden.get(table, [])}
        out[table.lower()] = {c.lower() for c in meta["columns"]} - blocked
    return out


def is_column_readable(table: str, column: str) -> bool:
    return column.lower() in _readable_columns().get(table.lower(), set())


class GuardedConnection(sqlite3.Connection):
    """sqlite3.Connection that remembers WHY the authorizer last denied
    something: "write", "restricted" (S3) or "function" (S5). Lets the
    validation node decide refuse-vs-retry without parsing error strings."""
    last_denial: str | None = None


def _make_authorizer(conn: GuardedConnection, real_tables: set[str]):
    readable = _readable_columns()

    def deny(kind):
        conn.last_denial = kind
        return sqlite3.SQLITE_DENY

    def authorizer(action, arg1, arg2, db_name, trigger):
        if action not in _ALLOWED_ACTIONS:
            return deny("write")                        # writes, PRAGMA, ATTACH ...
        if action == sqlite3.SQLITE_READ:               # arg1=table, arg2=column
            table = (arg1 or "").lower()
            # CTE / subquery names are not real tables: their data comes from
            # real tables, which are checked when they are read.
            if table not in real_tables and not table.startswith("sqlite_"):
                return sqlite3.SQLITE_OK
            if table not in readable:
                return deny("restricted")               # sqlite_master, undocumented tables
            if (arg2 or "") == "" or arg2.lower() in readable[table]:
                return sqlite3.SQLITE_OK                # "" = COUNT(*) style table read
            return deny("restricted")                   # hidden / undocumented column
        if action == sqlite3.SQLITE_FUNCTION:           # arg2=function name
            if (arg2 or "").lower() not in _ALLOWED_FUNCTIONS:
                return deny("function")
        return sqlite3.SQLITE_OK

    return authorizer


#establishes the first connection, checks if db exists in memory(RAM which gets lost if connection closes
#) or an actual path, and once path is set, you seed the dummy data
def get_connection() -> sqlite3.Connection:

    global _connection
    with _connection_lock:
        if _connection is None:
            is_new = DB_PATH == ":memory:" or not os.path.exists(DB_PATH)
            _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
            if is_new:
                _seed_dummy_data(_connection)
    return _connection


def get_readonly_connection() -> sqlite3.Connection:
    """S6: a NEW read-only connection per call -- no shared connection across
    concurrent API requests. Prefer the readonly_connection() context manager,
    which also closes it."""
    get_connection()  # make sure the file exists / is seeded first
    if DB_PATH == ":memory:":
        raise RuntimeError("Read-only mode needs a file DB; set DB_PATH to a .db file.")
    uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, factory=GuardedConnection)
    conn.execute("PRAGMA query_only = ON")      # must run BEFORE the authorizer (it denies PRAGMA)
    real_tables = {r[0].lower() for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    conn.set_authorizer(_make_authorizer(conn, real_tables))
    return conn


@contextmanager
def readonly_connection():
    conn = get_readonly_connection()
    try:
        yield conn
    finally:
        conn.close()


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
