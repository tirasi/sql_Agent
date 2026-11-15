"""
Turns the schema catalog (catalog.py) and schema_data/*.json into text an
LLM prompt can consume, and finds which tables / definitions a question
mentions.
"""

import json
import os
import re
from functools import lru_cache
from app.config import SCHEMA_DATA_DIR
from app.schema.catalog import get_catalog
from app.database.connection import is_column_readable, readonly_connection

_EXAMPLES_PER_COLUMN = 3
_EXAMPLE_MAX_CHARS = 40
_INSTRUCTION_LIKE = re.compile(
    r"(ignore|disregard|forget)\b.{0,40}\b(instruction|previous|above|rules)|you are now|system prompt",
    re.IGNORECASE)


def _load(filename: str, default=None):
    path = os.path.join(SCHEMA_DATA_DIR, filename)
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


# ---------- tables ----------

def get_all_table_names() -> list[str]:
    return list(get_catalog().keys())


def get_table_catalog_summary(table_names: list[str] | None = None) -> str:
    """Short summary of tables -- used by schema_retrieval's router."""
    cat = get_catalog()
    names = table_names or list(cat)
    return "\n".join(f"- {n}: {cat[n]['description']} (columns: {', '.join(cat[n]['columns'])})"
                     for n in names if n in cat)


def get_full_schema_context(table_names: list[str]) -> str:
    """Column-level detail for the chosen tables + only the business rules
    that apply to them (C5)."""
    cat = get_catalog()
    lines = []
    for name in table_names:
        if name not in cat:
            continue
        lines.append(f"TABLE {name}: {cat[name]['description']}")
        examples = column_examples(name)
        for col, col_desc in cat[name]["columns"].items():
            if is_column_readable(name, col):          # S3
                ex = examples.get(col)
                ex_txt = f"  (e.g. {', '.join(repr(v) for v in ex)})" if ex else ""
                lines.append(f"  - {col}: {col_desc}{ex_txt}")
    rules = get_rules(table_names)
    if rules:
        lines.append("\nBUSINESS RULES:")
        lines.extend(f"  - {r['description']}" for r in rules)
    return "\n".join(lines)


@lru_cache(maxsize=256)
def column_examples(table: str) -> dict[str, list[str]]:
    """A few real stored values per column, so the model sees FORMATS:
    a date column holding '201201' is YYYYMM, not YYYY-MM-DD (BIRD q1479
    failed on exactly this). Read through the read-only, allow-listed
    connection, so hidden columns (S3) are never sampled; values are short
    and instruction-like text is skipped (S4)."""
    out: dict[str, list[str]] = {}
    cols = [c for c in get_catalog().get(table, {}).get("columns", {}) if is_column_readable(table, c)]
    try:
        with readonly_connection() as conn:
            for col in cols:
                vals = []
                for (v,) in conn.execute(
                        f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 10'):
                    if isinstance(v, (bytes, bytearray)):
                        continue
                    v = " ".join(str(v).split())
                    if not v or _INSTRUCTION_LIKE.search(v):
                        continue
                    vals.append(v if len(v) <= _EXAMPLE_MAX_CHARS else v[:_EXAMPLE_MAX_CHARS] + "…")
                    if len(vals) == _EXAMPLES_PER_COLUMN:
                        break
                out[col] = vals
    except Exception:          # sampling is a hint only; never fail the request on it
        pass
    return out


# ---------- C6: table names in the question, by word forms ----------

def _plural(singular: str) -> str:
    if singular.endswith("y") and len(singular) > 1 and singular[-2] not in "aeiou":
        return singular[:-1] + "ies"
    if singular.endswith(("s", "x", "ch", "sh")):
        return singular + "es"
    return singular + "s"


def word_forms(word: str) -> set[str]:
    """order -> {order, orders, ordered, ordering}; category -> {category,
    categories, ...}; statuses -> {status, statuses, ...}. Replaces the old
    `rstrip("s") + \\w*` trick, which turned `status` into `statu` and matched
    any word that merely STARTED with a table name."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        singulars = {w[:-3] + "y"}
    elif w.endswith("es"):
        singulars = {w[:-2], w[:-1]}          # statuses -> status ; houses -> house
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        singulars = {w[:-1]}
    else:
        singulars = {w}
    forms = {w}
    for singular in singulars:
        forms |= {singular, _plural(singular)}
        if len(singular) > 3:
            stem = singular[:-1] if singular.endswith("e") else singular
            forms |= {stem + "ed", stem + "ing"}
    return forms


# English phrases that contain a table word but don't mean the table
_IDIOMS = re.compile(r"\bin order (?:to|of|for)\b|\border by\b", re.IGNORECASE)


def tables_named(question: str, known: list[str] | None = None) -> list[str]:
    text = _IDIOMS.sub(" ", question)
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = []
    for t in known or get_all_table_names():
        parts = [t] + t.split("_")
        if any(word_forms(p) & words for p in parts if len(p) > 2):
            hits.append(t)
    return hits


def top_tables(question: str, k: int) -> list[str]:
    """C10: on big schemas, rank tables by word overlap with the question
    (name, description, column names) and keep the top k for the router."""
    names = get_all_table_names()
    if len(names) <= k:
        return names
    cat = get_catalog()
    q = set(re.findall(r"[a-z]+", question.lower()))
    q_forms = set().union(*(word_forms(w) for w in q)) if q else set()

    def score(t):
        vocab = set(re.findall(r"[a-z]+", (t + " " + cat[t]["description"] + " " + " ".join(cat[t]["columns"])).lower()))
        return (len(word_forms(t) & q_forms) * 5) + len(vocab & q_forms)
    return sorted(names, key=score, reverse=True)[:k]


# ---------- C4: synonyms vs metrics ----------

def _synonym_file() -> dict:
    data = _load("synonyms.json", {}) or {}
    if "values" in data or "tables" in data:
        return data
    return {"values": data, "tables": {}}          # old flat format


def get_synonyms() -> dict:
    """Value + table synonyms, flattened (text that is safe to rewrite)."""
    d = _synonym_file()
    return {**d.get("values", {}), **d.get("tables", {})}


def get_metrics() -> dict:
    return _load("metrics.json", {}) or {}


def find_metrics(question: str) -> list[dict]:
    """Business terms that mean a calculation ("sales" = SUM(orders.amount)).
    These are NOT rewritten in the question; their exact SQL is handed to the
    SQL prompt instead."""
    hits = []
    for name, m in get_metrics().items():
        for alias in sorted(m.get("aliases", [name]), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", question, re.IGNORECASE):
                hits.append({"name": name, "term": alias, "sql": m["sql"],
                             "tables": m.get("tables", []), "description": m.get("description", "")})
                break
    return hits


# ---------- C5: rules scoped to tables, optionally executable ----------

def get_rules(table_names: list[str] | None = None) -> list[dict]:
    rules = _load("business_rules.json", []) or []
    if table_names is None:
        return rules
    chosen = {t.lower() for t in table_names}
    return [r for r in rules if not r.get("tables") or {t.lower() for t in r["tables"]} <= chosen]


def check_rules(question: str, sql: str, table_names: list[str]) -> str | None:
    """Runs the machine-checkable part of each applicable rule. Returns the
    first violation message, or None."""
    for r in get_rules(table_names):
        c = r.get("check")
        if not c:
            continue
        if c.get("when_question") and not re.search(c["when_question"], question, re.IGNORECASE):
            continue
        if c.get("when_sql") and not re.search(c["when_sql"], sql, re.IGNORECASE):
            continue
        if c.get("sql_must_match") and not re.search(c["sql_must_match"], sql, re.IGNORECASE):
            return f"Business rule '{r['rule']}' violated: {c.get('message', r['description'])}"
        if c.get("sql_must_not_match") and re.search(c["sql_must_not_match"], sql, re.IGNORECASE):
            return f"Business rule '{r['rule']}' violated: {c.get('message', r['description'])}"
    return None
