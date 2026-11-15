"""
Turns the schema catalog (catalog.py) and schema_data/*.json into text an
LLM prompt can consume, and finds which tables / definitions a question
mentions.
"""

import json
import os
import re
from app.config import SCHEMA_DATA_DIR
from app.schema.catalog import get_catalog
from app.database.connection import is_column_readable


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
        for col, col_desc in cat[name]["columns"].items():
            if is_column_readable(name, col):          # S3
                lines.append(f"  - {col}: {col_desc}")
    rules = get_rules(table_names)
    if rules:
        lines.append("\nBUSINESS RULES:")
        lines.extend(f"  - {r['description']}" for r in rules)
    return "\n".join(lines)


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
