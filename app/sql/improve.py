"""
Deterministic logic fixes for generated SQL (no LLM, no bigger model).

Small models make the same few LOGIC mistakes over and over, and retrying
the model does not fix them (BIRD smoke test: 3/3 identical wrong answers).
These are recognised in the parsed query (sqlglot AST) and rewritten:

1. count_condition     COUNT(Currency = 'EUR') counts every non-NULL row (it
                       is always the row count). -> SUM(CASE WHEN cond THEN 1 ELSE 0 END)
2. integer_division    SUM(..)/COUNT(..) on integers truncates in SQLite
                       (3/4 = 0). -> 1.0 * numerator / denominator
3. fanout_ranking      "Who had the least consumption in 2012?" ranked
                       customers by ONE monthly row (ORDER BY T2.Consumption
                       LIMIT 1). Ranking a parent (customer) by a child table
                       (yearmonth, many rows per customer) needs the TOTAL:
                       -> GROUP BY the selected columns, ORDER BY SUM(child col).
                       Skipped when the question asks about a single row
                       ("single", "individual", "one order", ...).
4. period_grouping     "Which year ...?" grouped by the full date (201201 = a
                       MONTH). If the question says year/month and the column's
                       stored values are longer dates, group by SUBSTR(col,...).
8. aggregate_in_case   CASE WHEN city='Jaipur' THEN COUNT(*) ELSE 0 END ... GROUP BY city
                       (aggregate INSIDE the CASE: one mostly-zero row per city).
                       -> SUM(CASE WHEN city='Jaipur' THEN 1 ELSE 0 END); rule 6 then
                       drops the GROUP BY.
9. ratio_completion    "What is the ratio of A to B?" but the query returns two counts
                       and no ratio. -> 1.0 * A / B, with A/B ordered as the question
                       mentions them.
6. conditional_groupby GROUP BY Currency + SUM(CASE WHEN Currency='CZK' ...) - SUM(CASE WHEN
                       Currency='EUR' ...): the CASE already splits by currency,
                       so grouping by it (without outputting it) breaks the
                       result into one row per currency (BIRD q1476). -> GROUP BY removed.
7. derived_projection  "What was the difference ...?" returned CZK total, EUR total
                       AND the difference. When one output is computed from the
                       others (a - b, a / b ...), the components are dropped.
5. ranking_projection  "Which year ... the most ...?" returned the year AND the
                       total it was ranked by. A who/which + most/least
                       question asks for the identity only: aggregate columns
                       used only for ranking are dropped from SELECT.

Every rule only fires on a clear pattern; anything unparseable is left
alone. Callers validate the result and fall back to the original SQL.
"""

import re
import sqlglot
from sqlglot import exp

from app.schema.catalog import get_catalog, get_relationships

_AGG = (exp.Sum, exp.Count, exp.Avg, exp.Max, exp.Min)
_PRED = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Is,
         exp.And, exp.Or, exp.Not, exp.Between)
_SINGLE_ROW_WORDS = re.compile(
    r"\b(single|individual|one (?:order|transaction|purchase|month|record|row)|"
    r"per (?:order|transaction|purchase|record)|a (?:single )?(?:order|transaction|purchase)|"
    r"largest (?:order|transaction|purchase|payment)|biggest (?:order|transaction|purchase|payment)|"
    r"smallest (?:order|transaction|purchase|payment))\b", re.IGNORECASE)
_SUPERLATIVE = re.compile(r"\b(most|least|highest|lowest|top|bottom|max(?:imum)?|min(?:imum)?|largest|"
                          r"smallest|biggest|best|worst|fewest|greatest|more|less)\b", re.IGNORECASE)
_IDENTITY_Q = re.compile(r"^\s*(?:in [^,]+,\s*)?(which|who|whose|what (?:is|was|are|were) the (?:name|id|year|month|date))\b",
                         re.IGNORECASE)


# ---------------------------------------------------------------- helpers

def _tables_in(select: exp.Select) -> dict[str, str]:
    """alias/name (lower) -> real table name, for this SELECT's FROM/JOINs."""
    out = {}
    for t in select.find_all(exp.Table):
        if t.find_ancestor(exp.Select) is not select:
            continue
        out[t.name.lower()] = t.name
        if t.alias:
            out[t.alias.lower()] = t.name
    return out


def _table_of(col: exp.Column, tables: dict[str, str]) -> str | None:
    if col.table:
        return tables.get(col.table.lower())
    cat = get_catalog()
    owners = {real for real in tables.values()
              if col.name.lower() in {c.lower() for c in cat.get(real, {}).get("columns", {})}}
    return owners.pop() if len(owners) == 1 else None


def _is_many_to_one(child: str, parents: set[str]) -> bool:
    """child has a foreign key to one of parents (child = many side)."""
    child_l, parents_l = child.lower(), {p.lower() for p in parents}
    for r in get_relationships():
        if r["from_table"].lower() == child_l and r["to_table"].lower() in parents_l:
            return True
    # no declared FK: same-named key column in both tables (e.g. CustomerID)
    cat = get_catalog()
    child_cols = {c.lower() for c in cat.get(child, {}).get("columns", {})}
    for p in parents:
        pcols = {c.lower() for c in cat.get(p, {}).get("columns", {})}
        if any(c.endswith("id") for c in child_cols & pcols):
            return True
    return False


def _date_prefix_len(table: str, column: str, unit: str) -> int | None:
    """How many leading chars of the stored value give the year/month."""
    from app.schema.retriever import column_examples   # local: avoids an import cycle
    vals = column_examples(table).get(column) or []
    if not vals:
        return None
    v = str(vals[0])
    if unit == "year":
        return 4 if re.match(r"^(19|20)\d{2}\D?\d{2}", v) and len(v) > 4 else None
    if re.match(r"^(19|20)\d{2}-\d{2}-\d{2}", v):
        return 7                                        # YYYY-MM-DD -> YYYY-MM
    if re.match(r"^(19|20)\d{6}$", v):
        return 6                                        # YYYYMMDD   -> YYYYMM
    return None


def _substr(col: exp.Expression, n: int) -> exp.Expression:
    return exp.Anonymous(this="SUBSTR", expressions=[col.copy(), exp.Literal.number(1), exp.Literal.number(n)])


# ---------------------------------------------------------------- rules

def _count_condition(tree: exp.Expression, notes: list) -> None:
    fixed = 0
    for c in list(tree.find_all(exp.Count)):
        arg = c.this
        if isinstance(arg, _PRED):
            c.replace(exp.Sum(this=exp.Case(ifs=[exp.If(this=arg.copy(), true=exp.Literal.number(1))],
                                            default=exp.Literal.number(0))))
            fixed += 1
    if fixed:
        notes.append(f"count_condition: {fixed} x COUNT(condition) -> SUM(CASE WHEN condition THEN 1 ELSE 0 END)")


def _integer_division(tree: exp.Expression, notes: list) -> None:
    for d in list(tree.find_all(exp.Div)):
        left, right = d.this, d.expression
        if not (left.find(exp.Sum, exp.Count) and right.find(exp.Sum, exp.Count)):
            continue
        if left.find(exp.Cast) or any(isinstance(l, exp.Literal) and "." in l.this for l in left.find_all(exp.Literal)):
            continue
        d.set("this", exp.Mul(this=exp.Literal.number("1.0"), expression=left.copy()))
        notes.append("integer_division: SUM/COUNT ratio made decimal (1.0 * ...)")


def _fanout_ranking(select: exp.Select, question: str, notes: list) -> None:
    if select.args.get("group") or not select.args.get("limit") or _SINGLE_ROW_WORDS.search(question):
        return
    order = select.args.get("order")
    if not order or len(order.expressions) != 1:
        return
    ordered = order.expressions[0]
    key = ordered.this
    if not isinstance(key, exp.Column) or any(isinstance(e.unalias(), _AGG) or e.find(*_AGG) for e in select.selects):
        return
    tables = _tables_in(select)
    sel_cols = [e.unalias() for e in select.selects]
    if not all(isinstance(c, exp.Column) for c in sel_cols):
        return
    parents = {_table_of(c, tables) for c in sel_cols}
    child = _table_of(key, tables)
    if None in parents or child is None or child in parents or not _is_many_to_one(child, parents):
        return
    agg = exp.Avg if re.search(r"\baverage\b|\bavg\b|\bmean\b", question, re.I) else exp.Sum
    select.set("group", exp.Group(expressions=[c.copy() for c in sel_cols]))
    ordered.set("this", agg(this=key.copy()))
    notes.append(f"fanout_ranking: ranked {'/'.join(sorted(parents))} by one {child} row -> "
                 f"GROUP BY + ORDER BY {agg.__name__.upper()}({key.sql('sqlite')})")


def _period_grouping(select: exp.Select, question: str, notes: list) -> None:
    unit = "year" if re.search(r"\b(year|yearly|annual(?:ly)?)\b", question, re.I) else \
           "month" if re.search(r"\b(month|monthly)\b", question, re.I) else None
    group = select.args.get("group")
    if not unit or not group:
        return
    tables = _tables_in(select)
    for g in list(group.expressions):
        if not isinstance(g, exp.Column):
            continue
        table = _table_of(g, tables)
        n = table and _date_prefix_len(table, g.name, unit)
        if not n:
            continue
        target = g.sql("sqlite")
        for node in list(select.find_all(exp.Column)):       # same column in SELECT / GROUP / ORDER
            if node.sql("sqlite") == target and node.find_ancestor(exp.Select) is select \
                    and not isinstance(node.parent, (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min)):
                node.replace(_substr(node, n))
        notes.append(f"period_grouping: grouped by full date {target} -> SUBSTR({target}, 1, {n}) ({unit})")


# the question ALSO asks for a number ("... and how much is the average price?"):
# the aggregate is part of the answer, not a ranking helper (BIRD 1531)
_ASKS_FOR_NUMBER = re.compile(r"\b(how (?:much|many)|average|avg|total|amount|sum|count|number of|price|cost|"
                              r"percentage|ratio)\b", re.IGNORECASE)


def _ranking_projection(select: exp.Select, question: str, notes: list) -> None:
    if not (_IDENTITY_Q.search(question) and _SUPERLATIVE.search(question)) or _ASKS_FOR_NUMBER.search(question):
        return
    if not (select.args.get("order") and select.args.get("limit")):
        return
    keep, drop = [], []
    for e in select.selects:
        (drop if isinstance(e.unalias(), _AGG) else keep).append(e)
    if not drop or not keep:
        return
    aliases = {e.alias.lower(): e.unalias() for e in drop if e.alias}
    for o in select.args["order"].expressions:
        if isinstance(o.this, exp.Column) and not o.this.table and o.this.name.lower() in aliases:
            o.set("this", aliases[o.this.name.lower()].copy())
    select.set("expressions", keep)
    notes.append("ranking_projection: dropped ranking-only column(s) "
                 + ", ".join(e.sql("sqlite") for e in drop) + " from SELECT")


def _aggregate_in_case(tree: exp.Expression, notes: list) -> None:
    fixed = 0
    for case in list(tree.find_all(exp.Case)):
        if case.find_ancestor(*_AGG) or len(case.args.get("ifs") or []) != 1:
            continue
        branch = case.args["ifs"][0]
        agg, default = branch.args.get("true"), case.args.get("default")
        if not isinstance(agg, _AGG):
            continue
        if default is not None and not (isinstance(default, exp.Null) or
                                        (isinstance(default, exp.Literal) and default.this in ("0", "0.0"))):
            continue
        cond = branch.this.copy()
        inner = agg.this
        if isinstance(agg, exp.Count) and (inner is None or isinstance(inner, exp.Star)):
            new = exp.Sum(this=exp.Case(ifs=[exp.If(this=cond, true=exp.Literal.number(1))],
                                        default=exp.Literal.number(0)))
        elif isinstance(agg, exp.Sum):
            new = exp.Sum(this=exp.Case(ifs=[exp.If(this=cond, true=inner.copy())], default=exp.Literal.number(0)))
        else:   # COUNT(x), AVG, MAX, MIN: rows outside the condition become NULL and are ignored
            new = type(agg)(this=exp.Case(ifs=[exp.If(this=cond, true=inner.copy())]))
        case.replace(new)
        fixed += 1
    if fixed:
        notes.append(f"aggregate_in_case: {fixed} x CASE WHEN cond THEN AGG(...) -> AGG(CASE WHEN cond ...)")


_RATIO_Q = re.compile(r"\bratio\b|\btimes as (?:many|much)\b|\btimes (?:more|bigger|larger)\b", re.IGNORECASE)


def _ratio_completion(select: exp.Select, question: str, notes: list) -> None:
    if not _RATIO_Q.search(question) or len(select.selects) != 2 or select.find(exp.Div):
        return
    a, b = (e.unalias() for e in select.selects)
    if not (isinstance(a, _AGG) and isinstance(b, _AGG)):
        return

    def first_mention(e):
        pos = [question.lower().find(str(l.this).lower()) for l in e.find_all(exp.Literal) if l.is_string]
        pos = [p for p in pos if p >= 0]
        return min(pos) if pos else None
    pa, pb = first_mention(a), first_mention(b)
    if pa is not None and pb is not None and pb < pa:
        a, b = b, a                                    # numerator = the one the question names first
    ratio = exp.Div(this=exp.Mul(this=exp.Literal.number("1.0"), expression=a.copy()), expression=b.copy())
    select.set("expressions", [exp.alias_(ratio, "ratio")])
    notes.append("ratio_completion: question asks for a ratio; replaced the two counts with 1.0 * A / B")


def _conditional_groupby(select: exp.Select, notes: list) -> None:
    group = select.args.get("group")
    if not group or not all(isinstance(g, exp.Column) for g in group.expressions):
        return
    outputs = {e.unalias().sql("sqlite").lower() for e in select.selects if isinstance(e.unalias(), exp.Column)}
    cond_cols = set()
    for agg in select.find_all(*_AGG):
        for branch in agg.find_all(exp.If):              # CASE WHEN ... and IIF(...) both parse to If
            for col in branch.this.find_all(exp.Column):
                cond_cols.add(col.sql("sqlite").lower())
    gcols = [g.sql("sqlite").lower() for g in group.expressions]
    if gcols and all(g in cond_cols and g not in outputs for g in gcols):
        select.set("group", None)
        notes.append(f"conditional_groupby: removed GROUP BY {', '.join(gcols)} (already split by CASE, not in output)")


_DERIVED_Q = re.compile(r"\b(difference|differ|ratio|percent(?:age)?|proportion|share|times|"
                        r"how much (?:more|less|higher|lower|bigger|smaller))\b", re.IGNORECASE)


def _derived_projection(select: exp.Select, question: str, notes: list) -> None:
    if not _DERIVED_Q.search(question) or len(select.selects) < 2:
        return
    def core(e):
        e = e.unalias()
        while isinstance(e, exp.Paren):
            e = e.this
        return e
    derived = [e for e in select.selects if isinstance(core(e), (exp.Sub, exp.Div, exp.Add, exp.Mul))
               and core(e).find(*_AGG)]
    if len(derived) != 1:
        return
    inside = {n.sql("sqlite").lower() for n in derived[0].find_all(exp.Expression)}
    drop = [e for e in select.selects if e is not derived[0] and core(e).sql("sqlite").lower() in inside]
    if not drop or len(drop) != len(select.selects) - 1:
        return                                           # only when EVERY other column is a component
    select.set("expressions", [derived[0]])
    notes.append("derived_projection: kept only the computed column; dropped components "
                 + ", ".join(e.alias_or_name for e in drop))


# ---------------------------------------------------------------- entry point

def improve_sql(sql: str, question: str) -> tuple[str, list[str]]:
    """Returns (possibly rewritten SQL, list of notes). Unchanged SQL and no
    notes when nothing applies or the SQL can't be parsed."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql, []
    if not isinstance(tree, exp.Select):
        return sql, []
    notes: list[str] = []
    try:
        _count_condition(tree, notes)
        _aggregate_in_case(tree, notes)
        _integer_division(tree, notes)
        _conditional_groupby(tree, notes)
        _ratio_completion(tree, question, notes)
        _derived_projection(tree, question, notes)
        _fanout_ranking(tree, question, notes)
        _period_grouping(tree, question, notes)
        _ranking_projection(tree, question, notes)
        # projections may have removed a column that was only grouped on (BIRD 1528:
        # Segment dropped, GROUP BY Segment left behind) -> check the GROUP BY again
        before = len(notes)
        _conditional_groupby(tree, notes)
        if len(notes) > before and any(n.startswith("conditional_groupby") for n in notes[:before]):
            notes.pop()                                  # same rule already reported
    except Exception:
        return sql, []
    return (tree.sql(dialect="sqlite") if notes else sql), notes
