"""
Loads schema_data/*.json and formats it into text an LLM prompt can consume.
Kept separate from database/metadata.py deliberately: this file is about
*describing* the schema for a model, that file is about *inspecting* the
real DB structurally. Different purposes, different consumers.
"""

import json
import os
from app.config import SCHEMA_DATA_DIR
from app.database.connection import is_column_readable


def _load(filename: str):
    path = os.path.join(SCHEMA_DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def get_all_table_names() -> list[str]:
    tables = _load("tables.json")
    return list(tables.keys())


def get_table_catalog_summary() -> str:
    """Short summary of every table -- used by schema_retrieval to pick
    relevant tables before pulling full column detail."""
    tables = _load("tables.json")
    lines = [f"- {name}: {meta['description']}" for name, meta in tables.items()]
    return "\n".join(lines)


def get_full_schema_context(table_names: list[str]) -> str:
    """Full column-level detail for only the tables that matter, plus any
    business rules relevant to those tables -- used by sql_generation."""
    tables = _load("tables.json")
    business_rules = _load("business_rules.json")

    lines = []
    for name in table_names:
        if name not in tables:
            continue
        meta = tables[name]
        lines.append(f"TABLE {name}: {meta['description']}")
        for col, col_desc in meta["columns"].items():
            if not is_column_readable(name, col):   # S3: hidden columns never reach the prompt
                continue
            lines.append(f"  - {col}: {col_desc}")

    if business_rules:
        lines.append("\nBUSINESS RULES:")
        for rule in business_rules:
            lines.append(f"  - {rule['description']}")

    return "\n".join(lines)


def get_synonyms() -> dict:
    return _load("synonyms.json")
