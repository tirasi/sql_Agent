"""
Value index: which table/column a literal value lives in.

Why: the small router model sometimes forgets a table ("How many customers
ordered Notebook?" -> picked only `customers`). But "Notebook" is a VALUE
stored in orders.product -- a fact we can look up deterministically instead
of hoping the model infers it.

Only low-cardinality text columns are indexed (<= MAX_DISTINCT distinct
values), e.g. product, city, name -- not free text or IDs. Built once and
cached.
"""

import re
from functools import lru_cache
from app.database.connection import get_connection

MAX_DISTINCT = 200


def _is_text_type(col_type: str) -> bool:
    t = (col_type or "").upper()
    return any(k in t for k in ("CHAR", "TEXT", "CLOB")) or t == ""


@lru_cache(maxsize=1)
def _build_index() -> dict:
    conn = get_connection()
    index: dict[str, set] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        for _, col, col_type, *_ in conn.execute(f'PRAGMA table_info("{t}")'):
            if not _is_text_type(col_type):
                continue
            n = conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{t}"').fetchone()[0]
            if n > MAX_DISTINCT:
                continue
            for (v,) in conn.execute(f'SELECT DISTINCT "{col}" FROM "{t}" WHERE "{col}" IS NOT NULL'):
                v = str(v).strip()
                # skip short strings and dates/numbers (no letters)
                if len(v) < 3 or not re.search(r"[A-Za-z]", v):
                    continue
                index.setdefault(v.lower(), set()).add((t, col, v))
    return index


def find_value_mentions(question: str) -> list[tuple[str, str, str]]:
    """Returns [(table, column, stored_value), ...] for values named in the question."""
    hits = []
    for key, locations in _build_index().items():
        if re.search(r"\b" + re.escape(key) + r"\b", question, re.IGNORECASE):
            hits.extend(sorted(locations))
    return hits
