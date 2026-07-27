"""
Database connection. SQLite for local dev/testing right now -- swap this
file's internals for Postgres/MySQL later (e.g. psycopg2/mysql-connector)
without touching any node or graph code, since everything above this layer
only calls get_connection() and executes plain SQL strings.
"""

import sqlite3
from app.config import DB_PATH


_connection: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Returns a singleton connection, creating + seeding the dummy DB on
    first call if it doesn't exist yet."""
    global _connection
    if _connection is None:
        is_new = DB_PATH == ":memory:" or not _db_file_exists()
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        if is_new:
            _seed_dummy_data(_connection)
    return _connection


def _db_file_exists() -> bool:
    import os
    return os.path.exists(DB_PATH)


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
