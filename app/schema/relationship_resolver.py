"""
Given a set of relevant tables, describes how they join -- handed to the
SQL-writing prompt so the model doesn't have to guess join columns.
Also finds "bridge" tables needed to connect selected tables.
"""

from collections import deque
from app.schema.catalog import get_relationships


def _load_relationships():
    # C10: real foreign keys from the DB + declared extras (catalog.py)
    return get_relationships()


def _neighbors() -> dict:
    graph: dict[str, set] = {}
    for r in _load_relationships():
        graph.setdefault(r["from_table"], set()).add(r["to_table"])
        graph.setdefault(r["to_table"], set()).add(r["from_table"])
    return graph


def connect_tables(tables: list[str]) -> list[str]:
    """If selected tables aren't directly related (e.g. customers + products
    with order_items in between), return the extra tables on the shortest
    foreign-key path (BFS) so a valid JOIN chain exists."""
    if len(tables) < 2:
        return []
    graph, extra = _neighbors(), []
    start = tables[0]
    for target in tables[1:]:
        prev, queue, seen = {}, deque([start]), {start}
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for nxt in graph.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = node
                    queue.append(nxt)
        node = target
        while node in prev:  # walk the path back
            node = prev[node]
            if node not in tables and node not in extra:
                extra.append(node)
    return extra


def get_join_context(table_names: list[str]) -> str:
    relevant = [
        r for r in _load_relationships()
        if r["from_table"] in table_names and r["to_table"] in table_names
    ]
    return "\n".join(
        f"  - {r['from_table']}.{r['from_column']} -> {r['to_table']}.{r['to_column']}"
        for r in relevant
    )
