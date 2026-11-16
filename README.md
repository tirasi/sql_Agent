# NLQ-to-SQL Agent

Ask a question in plain English, get a validated SQL query, the result, and an answer — from a
LangGraph agent that runs on small local models (Ollama) and **knows when not to trust itself**:
it compares several candidate queries, fixes common SQL logic mistakes in code, asks when a
question is genuinely ambiguous, and can never modify the database.

## Pipeline

```
question
  → intent_guard          refuse write requests ("delete …") before any LLM call
  → context_resolution    value/table synonyms, metric terms ("sales" = SUM(amount)), follow-ups
  → schema_retrieval      tables via LLM router + word forms + stored values + FK bridges
  → sql_generation        up to 3 candidate queries (plan line + interpretation each)
  → validation            read-only parse check, alias repair, logic fixes, business rules
  → execution             run all candidates, vote on results, confidence score
  → response_generation   answer with values filled in from the rows (no invented numbers)
     or clarify           "did you mean 1) … 2) …?" when readings disagree
```

Retries feed the specific error (plus a hint) back to the model; empty results are re-checked once.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
cp .env.example .env                  # LLM_PROVIDER=ollama (default) or anthropic
ollama pull phi4-mini && ollama pull qwen2.5-coder:3b && ollama pull qwen2.5:3b
python -m pytest tests -q             # offline tests, no LLM needed
```

## Use it

```bash
python app/ask.py                                   # interactive, on dummy.db
python app/ask.py "How many customers are there?"   # one question
python app/ask.py --db path/to/any.sqlite           # any SQLite file; schema read from the file
python app/ask.py --sql-only --candidates 1         # just the SQL, fastest
python app/server.py                                # web UI with step-by-step trace: http://127.0.0.1:8000
```

In `ask.py`: follow-up questions work, answer a clarifying question with its number,
`/tables`, `/reset`, `exit`.

## Measure it

```bash
python bench/run_benchmark.py --questions <mini_dev_sqlite.json> --db-root <dev_databases> --sample 100
python bench/show_results.py --failed               # SQL of the latest run
python bench/quick_eval.py                          # 23 questions on dummy.db (incl. safety)
```

Scores BIRD Mini-Dev by execution accuracy (strict, best guess, lenient). Every run is added to
[`bench/lookback.md`](bench/lookback.md): command, what changed, score, and each question's SQL and results.

## Project layout

```
app/
  ask.py, server.py, static/       command-line and web front ends
  graph/nlq_sql_graph.py           the LangGraph wiring and routing
  nodes/                           one file per graph node (see pipeline)
  sql/                             validator, executor (timeout, row cap), repair.py, improve.py (logic fixes)
  schema/                          catalog (live DB introspection), retriever, value index, bootstrap
  database/connection.py           read-only connection + authorizer (allow-listed tables, columns, functions)
  llm/client.py                    one call_llm() for Ollama/Anthropic, per-node model routing
  prompts/  state/  config.py
schema_data/                       descriptions, synonyms, metrics, business rules, access policy for dummy.db
bench/                             benchmark + eval scripts, lookback.md, results/
tests/                             test_security.py (S1–S9), test_pipeline.py (C1–C11 + logic fixes)
```

## Safety

Generated SQL runs on a read-only connection with a SQLite authorizer: only catalogued
tables/columns and allow-listed functions are readable, writes/PRAGMA/ATTACH are refused at
prepare time, each query has a timeout and row cap, and every request gets its own connection.
Database values are treated as data in prompts, and all values are escaped in the web UI.

## Configuration

All settings are environment variables — see `.env.example` (models, `SQL_CANDIDATES`,
`QUERY_TIMEOUT_MS`, `MAX_RESULT_ROWS`, `DEBUG_TRACE`, `REFERENCE_DATE`, `AUTO_EXPOSE_SCHEMA` …).

## License

MIT
