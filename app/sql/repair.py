"""
Deterministic SQL repairs -- fixes that need no LLM.

repair_alias: "no such column: T2.Consumption" where Consumption lives in the
table aliased T1. Small models often repeat this mistake on every retry even
when told which table has the column (BIRD q1479: 3 identical failures).
If exactly ONE table in the query has that column, the reference is
rewritten to that table's alias and the query is validated again.
"""

import re

_KEYWORDS = {"where", "on", "join", "inner", "left", "right", "full", "outer", "cross", "natural",
             "group", "order", "limit", "having", "union", "except", "intersect", "using", "as", "select"}
_FROM_JOIN = re.compile(r'\b(?:from|join)\s+("[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)(?:\s+(?:as\s+)?(\w+))?',
                        re.IGNORECASE)
_NO_COL = re.compile(r"no such column: (\w+)\.(\S+)")


def aliases(sql: str) -> dict[str, str]:
    """alias (lower) -> table name, for FROM/JOIN items (the table name is its own alias too)."""
    out = {}
    for m in _FROM_JOIN.finditer(sql):
        table = m.group(1).strip('"`[]')
        out.setdefault(table.lower(), table)
        alias = m.group(2)
        if alias and alias.lower() not in _KEYWORDS:
            out[alias.lower()] = table
    return out


def repair_alias(sql: str, error: str | None, table_columns: dict[str, set[str]]) -> str | None:
    """Returns the repaired SQL, or None if this isn't a fixable alias mix-up.
    table_columns: table (lower) -> column names (lower), e.g. catalog.readable_columns()."""
    m = _NO_COL.search(error or "")
    if not m:
        return None
    bad_alias, col = m.group(1), m.group(2).strip('"`[]')
    amap = aliases(sql)
    in_query = {t.lower() for t in amap.values()}
    owners = [t for t in in_query if col.lower() in table_columns.get(t, set())]
    if len(owners) != 1:
        return None                                  # 0 or several candidates: let the LLM decide
    good = [a for a, t in amap.items() if t.lower() == owners[0] and a != owners[0]] or [owners[0]]
    if len(good) != 1 or good[0] == bad_alias.lower():
        return None
    # keep the alias spelling used in the query
    good_alias = next(mm.group(2) for mm in _FROM_JOIN.finditer(sql)
                      if mm.group(2) and mm.group(2).lower() == good[0]) if good[0] != owners[0] else owners[0]
    pattern = re.compile(r"\b" + re.escape(bad_alias) + r'\.("?)' + re.escape(col) + r'\1(?!\w)', re.IGNORECASE)
    fixed = pattern.sub(lambda mm: f"{good_alias}.{mm.group(1)}{col}{mm.group(1)}", sql)
    return fixed if fixed != sql else None
