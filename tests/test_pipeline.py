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
BY_BIGGEST = "SELECT c.name FROM customers c ORDER BY c.name DESC LIMIT 1"                                                             # Ravi


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


# ---------- benchmark follow-ups (BIRD smoke test, 16 Nov) ----------
def test_schema_shows_example_values():
    ctx = retriever.get_full_schema_context(["orders"])
    assert "order_date: TEXT (YYYY-MM-DD)  (e.g. '2024-02-01'" in ctx


def test_date_context_does_not_assume_format():
    p = sg._date_context()
    assert "Dates are stored as TEXT in YYYY-MM-DD" not in p and "SUBSTR(col, 1, 4)" in p


def test_error_hint_points_to_owning_table():
    h = sg.error_hint("no such column: T2.amount")
    assert "orders" in h and "alias T2" in h
    assert "Available tables" in sg.error_hint("no such table: sales")


def test_retry_prompt_includes_hint():
    # a column no table has can't be repaired in code -> LLM retry gets a hint
    s = run("Total amount per customer?", [
        fence("SELECT c.name, sum(o.price) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id")] * 3 + [
        fence("SELECT c.name, sum(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id")])
    assert "no table has a column named price" in s["_prompts"][-1] and len(s["rows"]) == 4


def test_benchmark_lenient_match():
    sys.path.insert(0, str(ROOT / "bench"))
    from run_benchmark import lenient_match
    assert lenient_match([(1, 2, 0.5)], [(0.5,)])
    assert not lenient_match([(1, 2)], [(0.5,)])


def test_alias_mixup_repaired_without_llm():
    bad = ("SELECT SUBSTR(T1.order_date, 1, 4) AS y FROM orders AS T1 JOIN customers AS T2 "
           "ON T1.customer_id = T2.id WHERE T2.city = 'Jaipur' GROUP BY y ORDER BY SUM(T2.amount) DESC LIMIT 1")
    s = run("Which year had the most spend from Jaipur?", [fence(bad)] * 3)
    assert s["attempts"] == 1 and "SUM(T1.amount)" in s["sql_query"] and s["rows"] == [("2024",)]


def test_alias_repair_leaves_ambiguous_cases_alone():
    from app.sql.repair import repair_alias
    cols = {"a": {"id", "x"}, "b": {"id", "x"}}
    assert repair_alias("SELECT T3.x FROM a T1 JOIN b T2 ON T1.id = T2.id", "no such column: T3.x", cols) is None
    assert repair_alias('SELECT T2."x" FROM a AS T1 JOIN c AS T2 ON 1', "no such column: T2.x", {"a": {"x"}, "c": {"y"}}) == 'SELECT T1."x" FROM a AS T1 JOIN c AS T2 ON 1'


# ---------- deterministic logic fixes (app/sql/improve.py) ----------
from app.sql.improve import improve_sql   # noqa: E402


def test_fix_count_condition_and_integer_division():
    sql, notes = improve_sql("SELECT count(city = 'Jaipur') / count(city = 'Mumbai') AS r FROM customers", "ratio?")
    assert "SUM(CASE WHEN" in sql and "1.0 *" in sql and len(notes) == 2
    s = run("Ratio of Jaipur to Mumbai customers?", [fence("SELECT count(city = 'Jaipur') / count(city = 'Mumbai') AS r FROM customers")] * 3)
    assert s["rows"] == [(2.0,)]


def test_fix_fanout_ranking():
    q = "Which customer had the least spend?"
    bad = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY o.amount ASC LIMIT 1"
    sql, notes = improve_sql(bad, q)
    assert "GROUP BY c.name" in sql and "SUM(o.amount)" in sql
    assert run(q, [fence(bad)] * 3)["rows"] == [("Meera Shah",)]        # lowest TOTAL, not lowest single order


def test_fanout_not_applied_to_single_row_questions():
    bad = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY o.amount DESC LIMIT 1"
    assert improve_sql(bad, "Which customer placed the largest single order?")[1] == []


def test_fix_period_grouping_and_projection():
    q = "Which year had the most orders?"
    bad = "SELECT order_date, COUNT(*) AS n FROM orders GROUP BY order_date ORDER BY n DESC LIMIT 1"
    sql, notes = improve_sql(bad, q)
    assert "SUBSTR(order_date, 1, 4)" in sql and "COUNT(*) AS n" not in sql
    assert run(q, [fence(bad)] * 3)["rows"] == [("2024",)]


def test_projection_kept_when_question_wants_the_number():
    sql, notes = improve_sql("SELECT name, COUNT(*) AS n FROM customers GROUP BY name ORDER BY n DESC LIMIT 1",
                             "How many orders did the top customer place?")
    assert notes == []


def test_bird_debit_card_cases(tmp_path, monkeypatch):
    """The three failures from the BIRD smoke test, on a copy of their schema."""
    import sqlite3, json
    from app.schema import catalog as cat_mod, retriever as ret_mod
    db = tmp_path / "debit.sqlite"
    c = sqlite3.connect(db)
    c.executescript("""
      CREATE TABLE customers(CustomerID INTEGER PRIMARY KEY, Segment TEXT, Currency TEXT);
      CREATE TABLE yearmonth(CustomerID INTEGER, Date TEXT, Consumption REAL);
      INSERT INTO customers VALUES (1,'LAM','CZK'),(2,'LAM','EUR'),(3,'SME','CZK'),(4,'LAM','CZK');
      INSERT INTO yearmonth VALUES (1,'201201',5),(1,'201202',50),(2,'201201',20),(2,'201202',21),
                                   (4,'201201',30),(4,'201301',1),(3,'201301',100),(3,'201302',100);
    """)
    c.close()
    sd = tmp_path / "sd"; sd.mkdir()
    (sd / "tables.json").write_text(json.dumps({
        "customers": {"description": "c", "columns": {"CustomerID": "id", "Segment": "s", "Currency": "c"}},
        "yearmonth": {"description": "y", "columns": {"CustomerID": "id", "Date": "YYYYMM", "Consumption": "gas"}}}))
    monkeypatch.setattr(cat_mod, "DB_PATH", str(db))
    monkeypatch.setattr(cat_mod, "SCHEMA_DATA_DIR", str(sd))
    import app.database.connection as conn_mod
    monkeypatch.setattr(conn_mod, "DB_PATH", str(db))
    cat_mod.reset_cache(); ret_mod.column_examples.cache_clear(); conn_mod._readable_columns.cache_clear()
    try:
        # 1471: majority answer used COUNT(condition)
        sql, _ = improve_sql("SELECT count(Currency = 'EUR') / count(Currency = 'CZK') AS ratio_eur_czk FROM customers",
                             "What is the ratio of customers who pay in EUR against customers who pay in CZK?")
        assert sqlite3.connect(db).execute(sql).fetchall() == [(1 / 3,)]
        # 1472: ranked customers by one monthly row
        sql, _ = improve_sql("SELECT T1.CustomerID FROM customers AS T1 JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID "
                             "WHERE T1.Segment = 'LAM' AND T2.Date BETWEEN '201201' AND '201212' ORDER BY T2.Consumption ASC LIMIT 1",
                             "In 2012, who had the least consumption in LAM?")
        assert sqlite3.connect(db).execute(sql).fetchall() == [(4,)]       # total 30 (not customer 1's single 5)
        # 1479: grouped by month + returned the total
        sql, _ = improve_sql("SELECT T1.Date, SUM(T1.Consumption) AS TotalConsumption FROM yearmonth AS T1 JOIN customers AS T2 "
                             "ON T1.CustomerID = T2.CustomerID WHERE T2.Currency = 'CZK' GROUP BY T1.Date ORDER BY TotalConsumption DESC LIMIT 1",
                             "Which year recorded the most consumption of gas paid in CZK?")
        assert sqlite3.connect(db).execute(sql).fetchall() == [("2013",)]
    finally:
        cat_mod.reset_cache(); ret_mod.column_examples.cache_clear(); conn_mod._readable_columns.cache_clear()


def test_fix_conditional_groupby_and_derived_projection():
    q = "What was the difference in spend between Jaipur and Mumbai customers?"
    bad = ("SELECT SUM(CASE WHEN c.city = 'Jaipur' THEN o.amount ELSE 0 END) AS j, "
           "SUM(CASE WHEN c.city = 'Mumbai' THEN o.amount ELSE 0 END) AS m, "
           "(SUM(CASE WHEN c.city = 'Jaipur' THEN o.amount ELSE 0 END) - SUM(CASE WHEN c.city = 'Mumbai' THEN o.amount ELSE 0 END)) AS diff "
           "FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.city")
    sql, notes = improve_sql(bad, q)
    assert "GROUP BY" not in sql and " AS j" not in sql and len(notes) == 2
    assert run(q, [fence(bad)] * 3)["rows"] == [(590.0 - 1200.0,)]


def test_groupby_kept_when_column_is_output():
    sql, notes = improve_sql("SELECT c.city, SUM(CASE WHEN o.product = 'Notebook' THEN 1 ELSE 0 END) FROM customers c "
                             "JOIN orders o ON o.customer_id = c.id GROUP BY c.city", "notebooks per city")
    assert "GROUP BY" in sql


def test_sampling_noise_answers_instead_of_asking():
    # three single-reading replies that all disagree: noise, not ambiguity
    s = run("Who is the top customer?", [fence(BY_COUNT), fence(BY_SPEND), fence(BY_BIGGEST)])
    assert not s["needs_clarification"] and s["rows"] == [("Asha Patel",)]   # the temperature-0 candidate
    assert "1 of 3 candidate queries" in s["final_answer"]


# ---------- app/ask.py ----------
def test_ask_output_shows_sql_result_and_fixes():
    from app.ask import format_state
    bad = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY o.amount ASC LIMIT 1"
    s = run("Which customer had the least spend?", [fence(bad)] * 3, answer="{r1c1} spent the least.")
    text = format_state(s)
    assert "SQL:" in text and "GROUP BY" in text and "fixed in code" in text
    assert "Meera Shah" in text and "Answer: Meera Shah spent the least." in text
    assert "Result" not in format_state(s, sql_only=True)


def test_ask_shows_clarification_and_refusal():
    from app.ask import format_state
    assert "read-only" in format_state(run("Delete all orders", []))  or "cannot add" in format_state(run("Delete all orders", []))


def test_prose_reply_is_not_treated_as_sql():
    from app.ask import format_state
    s = run("How many customers are there?",
            ["I'm sorry, but I can't assist with that."] * 3 + [fence("SELECT count(*) FROM customers")])
    assert s["rows"] == [(4,)] and "contained no SQL query" in s["_prompts"][-1]
    never = run("asdf", ["I'm sorry, but I can't assist with that."] * 6)
    text = format_state(never)
    assert "SQL:" not in text and "Sorry, I couldn't answer" in text


def test_ask_rejects_pasted_commands():
    from app.ask import main
    with pytest.raises(SystemExit) as e:
        main(["python3", "app/ask.py", "--sql-only"])
    assert "looks like a command" in str(e.value)


# ---------- screenshot 16 Nov: "ratio of Jaipur customers to Mumbai customers" ----------
SCREENSHOT_SQL = ("SELECT CASE WHEN city = 'Jaipur' THEN COUNT(*) ELSE 0 END AS jaipur_count, "
                  "CASE WHEN city = 'Mumbai' THEN COUNT(*) ELSE 0 END AS mumbai_count FROM customers GROUP BY city")


def test_aggregate_in_case_and_ratio_completion():
    q = "What is the ratio of Jaipur customers to Mumbai customers?"
    sql, notes = improve_sql(SCREENSHOT_SQL, q)
    assert "GROUP BY" not in sql and "AS ratio" in sql and "1.0 *" in sql
    assert run(q, [fence(SCREENSHOT_SQL)] * 3, answer="The ratio is {r1c1}.")["rows"] == [(2.0,)]
    # order follows the question
    sql2, _ = improve_sql(SCREENSHOT_SQL, "What is the ratio of Mumbai customers to Jaipur customers?")
    import sqlite3
    assert sqlite3.connect("dummy.db").execute(sql2).fetchall() == [(0.5,)]


def test_llm_arithmetic_words_are_rejected():
    rows = [(0, 0), (2, 0), (0, 1)]
    assert not rg._numbers_are_grounded("There are twice as many Jaipur customers as Mumbai customers.", rows, "ratio?")
    assert not rg._numbers_are_grounded("There are four customers.", [(3,)], "how many?")
    assert rg._numbers_are_grounded("There are four customers.", [(4,)], "how many?")
    assert rg._numbers_are_grounded("Jaipur has twice as many.", [(2.0,)], "ratio?")      # computed by the SQL


# ---------- BIRD full-run analysis (38 questions, 16 Nov) ----------
def test_missing_join_added():
    # city is in customers, the query only reads orders (like BIRD 1521: Country in gasstations)
    s = run("How many orders came from Jaipur?", [fence("SELECT COUNT(*) FROM orders WHERE city = 'Jaipur'")] * 3)
    assert s["rows"] == [(3,)] and s["attempts"] == 1 and "JOIN customers" in s["sql_query"]


def test_aggregate_in_where_moved_to_having():
    bad = "SELECT c.name FROM customers AS c JOIN orders AS o ON o.customer_id = c.id WHERE COUNT(o.id) > 1"
    s = run("Which customers placed more than one order?", [fence(bad)] * 3)
    assert sorted(r[0] for r in s["rows"]) == ["Asha Patel", "Kabir Singh"] and "HAVING" in s["sql_query"]


def test_ranking_projection_keeps_asked_number():
    sql, notes = improve_sql("SELECT c.name, AVG(o.amount) AS avg_price FROM customers c JOIN orders o "
                             "ON o.customer_id = c.id GROUP BY c.name ORDER BY avg_price DESC LIMIT 1",
                             "Who is the top customer and how much is their average order?")
    assert "AVG(o.amount)" in sql.split("FROM")[0]


def test_groupby_rechecked_after_projection():
    # like BIRD 1528: component column dropped, its GROUP BY must go too
    bad = ("SELECT product, CAST(SUM(CASE WHEN product = 'Notebook' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS pct "
           "FROM orders WHERE customer_id = 4 GROUP BY product")
    s = run("What is the percentage of Notebook orders for customer 4?", [fence(bad)] * 3)
    assert s["rows"] == [(50.0,)]
