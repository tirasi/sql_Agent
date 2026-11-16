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


# ---------------------------------------------------------------- missing JOIN

def repair_missing_join(sql: str, error: str | None, catalog: dict, relationships: list) -> str | None:
    """'no such column: Country' when Country lives in a table the query did not
    join (BIRD 1509/1514/1521: 5 of 38 failures). If exactly one other table
    has the column and it can be joined to a table already in the query (a
    foreign key, or a same-named *ID column), the JOIN is added and the column
    qualified with it."""
    import sqlglot
    from sqlglot import exp
    m = re.search(r"no such column: ([A-Za-z_][\w ]*?)\s*$", error or "")
    if not m:
        return None
    col = m.group(1).strip()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    in_query = {}                                       # real table (lower) -> alias used in the query
    for t in tree.find_all(exp.Table):
        in_query[t.name.lower()] = t.alias or t.name
    cols = {t.lower(): {c.lower(): c for c in m_["columns"]} for t, m_ in catalog.items()}
    owners = [t for t in catalog if col.lower() in cols[t.lower()] and t.lower() not in in_query]
    if len(owners) != 1:
        return None
    new = owners[0]

    on = None
    for r in relationships:
        a, b = r["from_table"].lower(), r["to_table"].lower()
        if a == new.lower() and b in in_query:
            on = (in_query[b], r["to_column"], r["from_column"])
        elif b == new.lower() and a in in_query:
            on = (in_query[a], r["from_column"], r["to_column"])
        if on:
            break
    if not on:                                          # no declared FK: shared *ID column
        for qt, alias in in_query.items():
            shared = [c for c in cols.get(qt, {}) if c.endswith("id") and c in cols[new.lower()]]
            if shared:
                on = (alias, cols[qt][shared[0]], cols[new.lower()][shared[0]])
                break
    if not on:
        return None

    single_table = len(in_query) == 1
    base_alias = next(iter(in_query.values()))
    new_alias = new if new.lower() not in {a.lower() for a in in_query.values()} else new + "_j"
    for c in list(tree.find_all(exp.Column)):           # qualify: the moved column -> new table,
        if c.table:                                     # everything else -> the original table
            continue
        if c.name.lower() == col.lower():
            c.set("table", exp.to_identifier(new_alias))
        elif single_table and c.name.lower() in cols.get(next(iter(in_query)), {}):
            c.set("table", exp.to_identifier(base_alias))
    tree = tree.join(exp.to_table(new).as_(new_alias) if new_alias != new else exp.to_table(new),
                     on=f'{on[0]}."{on[1]}" = {new_alias}."{on[2]}"', join_type="inner")
    return tree.sql(dialect="sqlite")


# ---------------------------------------------------------------- aggregate in WHERE

def repair_where_aggregate(sql: str, error: str | None) -> str | None:
    """'misuse of aggregate function COUNT()': a condition like COUNT(x) > 20 was
    put in WHERE (BIRD 1323). It is moved to HAVING, grouped by the selected
    non-aggregate columns."""
    import sqlglot
    from sqlglot import exp
    if "misuse of aggregate" not in (error or ""):
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    where = tree.args.get("where") if isinstance(tree, exp.Select) else None
    if not where:
        return None
    aggs = (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)
    conj = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
    moved = [c for c in conj if c.find(*aggs)]
    kept = [c for c in conj if not c.find(*aggs)]
    if not moved:
        return None
    plain = [e.unalias() for e in tree.selects if not e.find(*aggs)]
    if not tree.args.get("group"):
        if not plain or not all(isinstance(p, exp.Column) for p in plain):
            return None
        tree.set("group", exp.Group(expressions=[p.copy() for p in plain]))
    tree.set("where", exp.Where(this=exp.and_(*kept)) if kept else None)
    having = exp.and_(*[m.copy() for m in moved])
    if tree.args.get("having"):
        having = exp.and_(tree.args["having"].this, having)
    tree.set("having", exp.Having(this=having))
    return tree.sql(dialect="sqlite")
