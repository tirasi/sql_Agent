"""
Regression tests for correctness fixes C1-C11. No Ollama needed: the LLM is
replaced by a scripted fake.  Run:  python -m pytest tests/test_pipeline.py -q
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_tmp = tempfile.mkdtemp(prefix="nlq_pipe_")
shutil.copy(ROOT / "dummy.db", os.path.join(_tmp, "t.db"))
os.environ.update(DB_PATH=os.path.join(_tmp, "t.db"), SQL_CANDIDATES="3", REFERENCE_DATE="2024-07-15")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pytest                                                         # noqa: E402
import app.nodes.schema_retrieval as sr                               # noqa: E402
import app.nodes.sql_generation as sg                                 # noqa: E402
import app.nodes.response_generation as rg                            # noqa: E402
from app.nodes.sql_generation import clean_sql, parse_candidates      # noqa: E402
from app.nodes.context_resolution import context_resolution_node, picked_option  # noqa: E402
from app.nodes.execution import same_answer                           # noqa: E402
from app.schema import retriever, catalog                             # noqa: E402
from app.graph.nlq_sql_graph import build_graph                       # noqa: E402
from app.state.agent_state import make_initial_state, make_turn       # noqa: E402

GRAPH = build_graph()


def fence(sql, interp=None):
    head = f"-- interpretation: {interp}\n" if interp else ""
    return f"{head}```sql\n{sql}\n```"


def run(question, sql_replies, answer="ok", history=None, router='["customers","orders"]'):
    """sql_replies: list of raw LLM replies, consumed in order."""
    replies = iter(sql_replies)
    prompts = []

    def fake_sql(system, user, node_name=None, temperature=0.0):
        prompts.append(system)
        return next(replies)
    with mock.patch.object(sr, "call_llm", return_value=router), \
         mock.patch.object(sg, "call_llm", side_effect=fake_sql), \
         mock.patch.object(rg, "call_llm", return_value=answer):
        state = GRAPH.invoke(make_initial_state(question, history))
    state["_prompts"] = prompts
    return state


# ---------- C1: majority vote over candidate RESULTS ----------
def test_c1_majority_beats_a_wrong_candidate():
    good = "SELECT name FROM customers WHERE city = 'Jaipur'"
    bad = "SELECT name FROM customers"                       # dropped the filter
    s = run("List customers from Jaipur", [fence(bad), fence(good), fence(good + " ORDER BY name")])
    assert sorted(r[0] for r in s["rows"]) == ["Asha Patel", "Meera Shah"]
    assert s["confidence"] == 0.67


def test_c1_empty_result_is_rechecked_once():
    s = run("How many orders for notebook?", [
        fence("SELECT count(*) FROM orders WHERE product = 'notebook' AND amount > 9999")] * 3 + [
        fence("SELECT count(*) FROM orders WHERE product = 'Notebook'")])
    # count(*) returns one row, so use a rows-returning query to test empties:
    s = run("Which orders were notebooks?", [
        fence("SELECT id FROM orders WHERE product = 'notebooks'")] * 3 + [
        fence("SELECT id FROM orders WHERE product = 'Notebook'")])
    assert len(s["rows"]) == 3 and s["attempts"] == 2
    assert "returned 0 rows" in s["_prompts"][-1]


def test_c1_second_empty_result_is_accepted():
    q = "SELECT name FROM customers WHERE city = 'Chennai'"
    s = run("Customers in Chennai?", [fence(q)] * 3 + [fence(q)])
    assert s["rows"] == [] and s["final_answer"].startswith("No matching records")


def test_c1_lenient_result_matching():
    assert same_answer([("Asha",), ("Meera",)], [("Meera", "Jaipur"), ("Asha", "Jaipur")])
    assert not same_answer([("Asha",)], [("Asha",), ("Meera",)])


# ---------- C2: numbers come from the rows, not the LLM ----------
def test_c2_placeholders_filled_with_exact_values():
    s = run("How many customers are there?", [fence("SELECT count(*) FROM customers")] * 3,
            answer="There are {r1c1} customers.")
    assert s["final_answer"] == "There are 4 customers."


def test_c2_invented_number_is_replaced():
    s = run("How many customers are there?", [fence("SELECT count(*) FROM customers")] * 3,
            answer="There are about 40 customers.")
    assert "40" not in s["final_answer"] and "4" in s["final_answer"]


def test_c2_bad_placeholder_falls_back():
    s = run("How many customers are there?", [fence("SELECT count(*) FROM customers")] * 3,
            answer="There are {r9c9} customers.")
    assert "{r9c9}" not in s["final_answer"]


# ---------- C3: robust SQL extraction ----------
@pytest.mark.parametrize("raw,expected", [
    ("Here is the query with a join: SELECT name FROM customers", "SELECT name FROM customers"),
    ("```sql\nSELECT 1 FROM customers;\n```\nThis query selects ...", "SELECT 1 FROM customers"),
    ("I will select the customers.\nSELECT name FROM customers\n\nThis returns names.", "SELECT name FROM customers"),
    ("WITH j AS (SELECT * FROM customers) SELECT count(*) FROM j", "WITH j AS (SELECT * FROM customers) SELECT count(*) FROM j"),
    ("DELETE FROM orders", "DELETE FROM orders"),
])
def test_c3_clean_sql(raw, expected):
    assert " ".join(clean_sql(raw).split()) == expected


def test_c3_multiple_interpretations_parsed():
    raw = (fence("SELECT 1", "by number of orders") + "\n" + fence("SELECT 2", "by total spend"))
    c = parse_candidates(raw)
    assert [x["interpretation"] for x in c] == ["by number of orders", "by total spend"]


# ---------- C4: "sales" is a metric, not a rewrite to "orders" ----------
def test_c4_sales_not_rewritten_and_metric_found():
    out = context_resolution_node(make_initial_state("Total sales in Mumbai"))
    assert out["resolved_input"] == "Total sales in Mumbai"
    assert out["metric_hints"][0]["sql"] == "SUM(orders.amount)"


def test_c4_metric_reaches_prompt_and_adds_table():
    s = run("Total sales in Mumbai", [fence("SELECT SUM(o.amount) FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.city = 'Mumbai'")] * 3,
            router='["customers"]')
    assert "SUM(orders.amount)" in s["_prompts"][0] and "orders" in s["relevant_tables"]


def test_c4_value_synonyms_still_rewritten():
    assert context_resolution_node(make_initial_state("How many notebooks?"))["resolved_input"] == "How many Notebook?"


# ---------- C5: rules scoped to tables and checked ----------
def test_c5_rules_only_for_chosen_tables():
    assert [r["rule"] for r in retriever.get_rules(["customers"])] == []
    assert "counting_customers" in [r["rule"] for r in retriever.get_rules(["customers", "orders"])]


def test_c5_rule_violation_triggers_retry():
    wrong = "SELECT count(*) FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.product = 'Notebook'"
    right = "SELECT count(DISTINCT c.id) FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.product = 'Notebook'"
    s = run("How many customers ordered Notebook?", [fence(wrong)] * 3 + [fence(right)])
    assert s["rows"] == [(3,)] and "counting_customers" in s["_prompts"][-1]


# ---------- C6: table names by word forms ----------
def test_c6_word_forms():
    assert retriever.tables_named("Who ordered a backpack?") == ["orders"]
    assert retriever.tables_named("list customers in order of signup") == ["customers"]
    assert "status" in retriever.word_forms("statuses") and "statu" not in retriever.word_forms("status")
    assert "categories" in retriever.word_forms("category")


# ---------- C7: today's date in the prompt ----------
def test_c7_date_context():
    s = run("Orders last month?", [fence("SELECT id FROM orders WHERE order_date >= date('2024-07-15','-1 month')")] * 3)
    assert "Today's date is 2024-07-15" in s["_prompts"][0] and "date('now')" in s["_prompts"][0]


# ---------- C8: follow-up questions ----------
def test_c8_follow_up_uses_previous_turn():
    first = run("Which customers are in Jaipur?", [fence("SELECT name FROM customers WHERE city = 'Jaipur'")] * 3)
    history = [make_turn(first)]
    s = run("What about Mumbai?", [fence("SELECT name FROM customers WHERE city = 'Mumbai'")] * 3,
            history=history, router="[]")
    assert s["is_follow_up"] and "customers" in s["relevant_tables"]
    assert "Previous SQL: SELECT name FROM customers WHERE city = 'Jaipur'" in s["_prompts"][0]
    assert s["rows"] == [("Ravi Kumar",)]


def test_c8_independent_question_is_not_follow_up():
    out = context_resolution_node(make_initial_state("How many customers are there?",
                                                     [{"question": "x", "sql": "SELECT 1", "relevant_tables": []}]))
    assert not out["is_follow_up"]


# ---------- C9: ambiguity -> clarifying question -> reuse picked SQL ----------
BY_COUNT = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY count(*) DESC, c.id LIMIT 1"   # Asha
BY_SPEND = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY sum(o.amount) DESC LIMIT 1"     # Kabir
BY_BIGGEST = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY o.amount DESC LIMIT 1"                       # Ravi


def test_c9_disagreement_asks_and_pick_reuses_sql():
    s = run("Who is the top customer?", [
        fence(BY_COUNT, "by number of orders") + "\n" + fence(BY_SPEND, "by total spend"),
        fence(BY_BIGGEST, "by biggest single order"),
        fence(BY_SPEND, "by total spend")])                    # duplicate -> de-duplicated
    assert s["needs_clarification"] and "more than one way" in s["final_answer"]
    # most-voted reading first (total spend was produced twice)
    assert [o["interpretation"] for o in s["clarification_options"]] == [
        "by total spend", "by number of orders", "by biggest single order"]
    picked = run("by total spend", [], history=[make_turn(s)])   # no SQL LLM call needed
    assert picked["rows"] == [("Kabir Singh",)] and picked["_prompts"] == []


def test_c9_majority_means_no_question():
    s = run("Who is the top customer?", [fence(BY_SPEND), fence(BY_SPEND + " "), fence(BY_COUNT)])
    assert not s["needs_clarification"]


def test_c9_option_matching():
    opts = [{"interpretation": "by number of orders"}, {"interpretation": "by total spend"}]
    assert picked_option("2", opts) == 1
    assert picked_option("the first one", opts) == 0
    assert picked_option("total spend", opts) == 1
    assert picked_option("how many customers are there", opts) is None


# ---------- C10: schema from the live DB ----------
def test_c10_catalog_from_db_and_fk_discovered(monkeypatch):
    cat = catalog.get_catalog()
    assert set(cat) == {"customers", "orders"}
    assert ("customer_id", "customers", "id") in cat["orders"]["fks"]


def test_c10_undocumented_table_hidden_and_drift_ignored(tmp_path, monkeypatch):
    import sqlite3, json
    db = tmp_path / "d.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE a(id INTEGER PRIMARY KEY, x TEXT); CREATE TABLE secret(id INTEGER, pw TEXT);")
    c.close()
    sd = tmp_path / "sd"; sd.mkdir()
    (sd / "tables.json").write_text(json.dumps({"a": {"description": "A", "columns": {"id": "pk", "x": "x", "gone": "dropped column"}}}))
    monkeypatch.setattr(catalog, "DB_PATH", str(db))
    monkeypatch.setattr(catalog, "SCHEMA_DATA_DIR", str(sd))
    catalog.reset_cache()
    try:
        cat = catalog.get_catalog()
        assert set(cat) == {"a"} and set(cat["a"]["columns"]) == {"id", "x"}
    finally:
        catalog.reset_cache()


# ---------- C11: graph built once ----------
def test_c11_server_builds_graph_once():
    import app.server as server
    with mock.patch.object(server, "build_graph") as bg:
        assert server.GRAPH is not None
        bg.assert_not_called()
