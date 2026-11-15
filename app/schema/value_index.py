"""
Value index: which table/column a literal value lives in.

Why: the small router model sometimes forgets a table ("How many customers
ordered Notebook?" -> picked only `customers`). But "Notebook" is a VALUE
stored in orders.product -- a fact we can look up deterministically instead
of hoping the model infers it.

Only low-cardinality text columns are indexed (<= MAX_DISTINCT distinct
values), e.g. product, city, name -- not free text or IDs. Built once and
cached.

S3: only columns the access policy allows are indexed, so hidden values
    never reach a prompt.
S4: values that could act as prompt text are skipped: long strings,
    multi-line strings, and anything that reads like an instruction
    ("ignore previous instructions ...").
"""

import re
from functools import lru_cache
import sqlite3
import os
from app.config import DB_PATH
from app.database.connection import get_connection, is_column_readable

MAX_DISTINCT = 200
MAX_VALUE_CHARS = 60
_INSTRUCTION_LIKE = re.compile(
    r"(ignore|disregard|forget)\b.{0,40}\b(instruction|previous|above|rules)|"
    r"you are now|system prompt|<\s*/?\s*(system|data|user|assistant)",
    re.IGNORECASE,
)


def _safe_value(v: str) -> bool:
    return (3 <= len(v) <= MAX_VALUE_CHARS and "\n" not in v
            and re.search(r"[A-Za-z]", v) is not None
            and not _INSTRUCTION_LIKE.search(v))


def _is_text_type(col_type: str) -> bool:
    t = (col_type or "").upper()
    return any(k in t for k in ("CHAR", "TEXT", "CLOB")) or t == ""


@lru_cache(maxsize=1)
def _build_index() -> dict:
    get_connection()  # ensure the DB exists / is seeded
    # own short-lived read-only connection (S6: no shared connection)
    conn = sqlite3.connect(f"file:{os.path.abspath(DB_PATH)}?mode=ro", uri=True)
    index: dict[str, set] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        for _, col, col_type, *_ in conn.execute(f'PRAGMA table_info("{t}")'):
            if not _is_text_type(col_type) or not is_column_readable(t, col):
                continue
            n = conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{t}"').fetchone()[0]
            if n > MAX_DISTINCT:
                continue
            for (v,) in conn.execute(f'SELECT DISTINCT "{col}" FROM "{t}" WHERE "{col}" IS NOT NULL'):
                v = str(v).strip()
                # skip short strings, dates/numbers, long/multi-line/instruction-like text
                if not _safe_value(v):
                    continue
                index.setdefault(v.lower(), set()).add((t, col, v))
    conn.close()
    return index


def find_value_mentions(question: str) -> list[tuple[str, str, str]]:
    """Returns [(table, column, stored_value), ...] for values named in the question."""
    hits = []
    for key, locations in _build_index().items():
        if re.search(r"\b" + re.escape(key) + r"\b", question, re.IGNORECASE):
            hits.extend(sorted(locations))
    return hits
