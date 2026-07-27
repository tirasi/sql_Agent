"""
Given a set of relevant tables, describes how they join -- handed to the
SQL-writing prompt so the model doesn't have to guess join columns.
"""

import json
import os
from app.config import SCHEMA_DATA_DIR


def _load_relationships():
    path = os.path.join(SCHEMA_DATA_DIR, "relationships.json")
    with open(path, "r") as f:
        return json.load(f)


def get_join_context(table_names: list[str]) -> str:
    relationships = _load_relationships()
    relevant = [
        r for r in relationships
        if r["from_table"] in table_names and r["to_table"] in table_names
    ]
    if not relevant:
        return ""

    lines = ["JOIN PATHS:"]
    for r in relevant:
        lines.append(
            f"  - {r['from_table']}.{r['from_column']} -> {r['to_table']}.{r['to_column']}"
        )
    return "\n".join(lines)
