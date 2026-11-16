"""
Prepare schema_data for ANY SQLite database (BIRD/Spider benchmark DBs, or
your own file): a tables.json listing every real table/column, with
descriptions taken from BIRD-style database_description/*.csv when present,
plus empty synonyms/metrics/rules. Used by bench/run_benchmark.py and app/ask.py.
"""

import csv
import json
import sqlite3
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                        for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    return []


def build_schema_dir(db_file: Path, out_dir: Path) -> None:
    """tables.json for one benchmark DB: every real table/column, with BIRD's
    database_description/*.csv text as descriptions when present."""
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    tables = {}
    desc_dir = db_file.parent / "database_description"
    csv_by_table = {p.stem.lower(): p for p in desc_dir.glob("*.csv")} if desc_dir.exists() else {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        cols = {r[1]: (r[2] or "untyped") for r in conn.execute(f'PRAGMA table_info("{t}")')}
        by_lower = {c.lower(): c for c in cols}
        if t.lower() in csv_by_table:
            for row in _read_csv(csv_by_table[t.lower()]):
                real = by_lower.get((row.get("original_column_name") or "").lower())
                if not real:
                    continue
                parts = [cols[real]]
                for key in ("column_description", "value_description"):
                    v = " ".join((row.get(key) or "").split())
                    if v and v.lower() not in ("nan", "none"):
                        parts.append(v[:300])
                cols[real] = " - ".join(parts)
        tables[t] = {"description": f"Table {t}.", "columns": cols}
    conn.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=1))
    for name, empty in (("relationships.json", []), ("business_rules.json", []),
                        ("synonyms.json", {"values": {}, "tables": {}}), ("metrics.json", {}),
                        ("access_policy.json", {"hidden_columns": {}})):
        (out_dir / name).write_text(json.dumps(empty))
