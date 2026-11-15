"""
Schema catalog (C10): the single source of truth for which tables, columns
and foreign keys exist.

Before: schema_data/tables.json and relationships.json were typed by hand and
silently drifted from the real DB; the pipeline never looked at the DB.
Now:
  - tables, columns, types and foreign keys are read from the LIVE database
    (PRAGMA table_info / foreign_key_list);
  - tables.json only ADDS human descriptions on top (a column in tables.json
    that no longer exists in the DB is ignored and logged as drift);
  - relationships.json only adds joins that are not declared as real FKs.
Security (S3) is kept: a table/column the DB has but tables.json doesn't
document stays hidden unless AUTO_EXPOSE_SCHEMA=true, and access_policy.json
can hide more columns.
"""

import json
import logging
import os
import sqlite3

from app.config import DB_PATH, SCHEMA_DATA_DIR, ACCESS_POLICY_FILE, AUTO_EXPOSE_SCHEMA

log = logging.getLogger("nlq_sql")
_cache: dict | None = None


def _load_json(name: str, default):
    path = os.path.join(SCHEMA_DATA_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _introspect() -> dict | None:
    """{table: {"columns": {col: type}, "fks": [(col, ref_table, ref_col)]}},
    or None if the DB file doesn't exist yet."""
    if DB_PATH == ":memory:" or not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(f"file:{os.path.abspath(DB_PATH)}?mode=ro", uri=True)
    try:
        out = {}
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            info = list(conn.execute(f'PRAGMA table_info("{t}")'))
            pk = next((r[1] for r in info if r[5]), "id")
            cols = {r[1]: (r[2] or "") for r in info}
            fks = []
            for r in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
                # r = (id, seq, table, from, to, ...); "to" is NULL when it means the PK
                ref_pk = next((x[1] for x in conn.execute(f'PRAGMA table_info("{r[2]}")') if x[5]), pk)
                fks.append((r[3], r[2], r[4] or ref_pk))
            out[t] = {"columns": cols, "fks": fks}
        return out
    finally:
        conn.close()


def get_catalog() -> dict:
    """{table: {"description": str, "columns": {col: description}, "fks": [...]}}
    -- only what generated SQL may see."""
    global _cache
    if _cache is not None:
        return _cache
    documented = _load_json("tables.json", {})
    hidden = {t.lower(): {c.lower() for c in cols}
              for t, cols in _load_json(ACCESS_POLICY_FILE, {}).get("hidden_columns", {}).items()}
    live = _introspect()
    from_db = live is not None
    if not from_db:   # DB not created yet: use the documented catalog, don't cache
        live = {t: {"columns": {c: "" for c in m["columns"]}, "fks": []} for t, m in documented.items()}

    catalog = {}
    for table, info in live.items():
        doc = documented.get(table)
        if doc is None and not AUTO_EXPOSE_SCHEMA:
            log.info("table %s exists in the DB but not in tables.json -> hidden", table)
            continue
        doc_cols = (doc or {}).get("columns", {})
        for col in doc_cols:
            if col not in info["columns"]:
                log.warning("schema drift: %s.%s is in tables.json but not in the database -- ignored", table, col)
        cols = {}
        for col, ctype in info["columns"].items():
            if col.lower() in hidden.get(table.lower(), set()):
                continue
            if col not in doc_cols and not AUTO_EXPOSE_SCHEMA:
                continue
            cols[col] = doc_cols.get(col) or ctype or "untyped"
        catalog[table] = {
            "description": (doc or {}).get("description", f"Table {table} (auto-discovered, no description)."),
            "columns": cols,
            "fks": info["fks"],
        }
    if from_db:
        _cache = catalog
    return catalog


def reset_cache() -> None:
    global _cache
    _cache = None


def readable_columns() -> dict[str, set[str]]:
    return {t.lower(): {c.lower() for c in m["columns"]} for t, m in get_catalog().items()}


def get_relationships() -> list[dict]:
    """Real foreign keys from the DB + extra joins from relationships.json,
    limited to visible tables and columns."""
    cat = get_catalog()
    visible = readable_columns()
    rels, seen = [], set()

    def add(ft, fc, tt, tc, kind):
        key = (ft.lower(), fc.lower(), tt.lower(), tc.lower())
        if key in seen:
            return
        if fc.lower() in visible.get(ft.lower(), set()) and tc.lower() in visible.get(tt.lower(), set()):
            seen.add(key)
            rels.append({"from_table": ft, "from_column": fc, "to_table": tt, "to_column": tc, "relationship": kind})

    for t, m in cat.items():
        for col, ref_t, ref_c in m["fks"]:
            add(t, col, ref_t, ref_c, "many_to_one")
    for r in _load_json("relationships.json", []):
        add(r["from_table"], r["from_column"], r["to_table"], r["to_column"], r.get("relationship", "declared"))
    return rels
