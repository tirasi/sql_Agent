"""
Central configuration, loaded from environment variables (.env).

LLM_PROVIDER now defaults to "ollama" (free, local) so the app actually
generates real SQL out of the box, instead of the "mock" stand-in used
during initial wiring/testing. The mock provider only pattern-matches a
couple of hardcoded phrases ("how many" + a product name) and returns a
generic fallback query otherwise -- it never reads schema_context or
join_context, so it will not produce correct SQL for anything beyond the
handful of cases it was written for. Switch back to "mock" only if you
want zero-setup testing of graph wiring with no LLM calls at all, or to
"anthropic" if you have API budget and want the strongest quality on
sql_generation.

Requirements for LLM_PROVIDER=ollama:
- Ollama must be installed and running locally (OLLAMA_HOST below).
- Each model in NODE_MODELS_OLLAMA must be pulled first, e.g.:
    ollama pull phi4-mini
    ollama pull qwen2.5-coder:7b
    ollama pull gemma3:4b
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Per-node model routing. Each graph node can use a different model sized
# to what that step actually needs -- classification-style steps get a
# small/fast model, code-generation gets a coding-tuned model, deterministic
# steps (validation/execution) never call an LLM at all.
#
# FREE / LOCAL (Ollama) -- what this file defaults to, tuned for 8GB RAM:
NODE_MODELS_OLLAMA = {
    "schema_retrieval":    "phi4-mini",        # fast classification: "which tables?"
    "sql_generation":      "qwen2.5-coder:3b", # coding-tuned, best free option for SQL syntax
    "response_generation": "qwen2.5:3b",        # plain-language summary, low stakes
}

# PAID (Anthropic) -- what to switch to if you have API budget / GPUs.
# Mixing tiers even here: cheap model for easy steps, strong model for the
# step that actually determines correctness (SQL generation).
NODE_MODELS_ANTHROPIC = {
    "schema_retrieval":    "claude-haiku-4-5-20251001",
    "sql_generation":      "claude-sonnet-4-6",
    "response_generation": "claude-haiku-4-5-20251001",
}

# --- Database ---
# SQLite dummy DB by default -- swap DB_PATH / connection.py for
# Postgres/MySQL later without touching any node logic above it.
DB_PATH = os.getenv("DB_PATH", "dummy.db")

# --- Retry policy ---
MAX_SQL_ATTEMPTS = int(os.getenv("MAX_SQL_ATTEMPTS", "3"))

# --- Schema data files ---
SCHEMA_DATA_DIR = os.getenv("SCHEMA_DATA_DIR", "schema_data")

# --- Safety limits (S1, S2, S7) ---
# S1: any single query is aborted after this many milliseconds (stops
#     endless WITH RECURSIVE loops and huge cross joins).
QUERY_TIMEOUT_MS = int(os.getenv("QUERY_TIMEOUT_MS", "3000"))
# S2: at most this many rows are fetched from the DB per query ...
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "500"))
# ... and at most this many are shown to the answer-writing LLM.
ANSWER_SAMPLE_ROWS = int(os.getenv("ANSWER_SAMPLE_ROWS", "30"))
# S7: when false, raw DB errors / attempt history / schema text are NOT sent
#     to the browser (they are still logged server-side). Turn on locally.
DEBUG_TRACE = os.getenv("DEBUG_TRACE", "false").lower() in ("1", "true", "yes")

# --- Access policy (S3) ---
# Only tables/columns described in schema_data/tables.json are readable by
# generated SQL; access_policy.json can hide more columns on top of that.
ACCESS_POLICY_FILE = os.getenv("ACCESS_POLICY_FILE", "access_policy.json")

# --- Answer quality (C1, C7, C8, C9, C10) ---
# C1: how many SQL-generation calls per question. Call 1 is deterministic
#     (temperature 0); the rest are sampled (temperature 0.7). Their RESULTS
#     are compared and the majority answer wins. 1 = old single-shot behaviour.
SQL_CANDIDATES = int(os.getenv("SQL_CANDIDATES", "3"))
# C9: if the candidate queries disagree with no majority, ask the user which
#     reading they meant instead of guessing.
CLARIFY_ON_DISAGREEMENT = os.getenv("CLARIFY_ON_DISAGREEMENT", "true").lower() in ("1", "true", "yes")
# C7: "today" for relative dates ("last month"). Empty = the real date.
#     Set e.g. REFERENCE_DATE=2024-07-15 to make evals reproducible.
REFERENCE_DATE = os.getenv("REFERENCE_DATE", "")
# C8: how many previous turns of a conversation are kept per session.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))
# C10: tables/columns found in the live DB but NOT documented in tables.json
#      are hidden unless this is true (security first: S3 allow-list).
AUTO_EXPOSE_SCHEMA = os.getenv("AUTO_EXPOSE_SCHEMA", "false").lower() in ("1", "true", "yes")
# C10: with more tables than this, only the best-matching ones are shown to
#      the router LLM, so the prompt stays small on big schemas.
ROUTER_MAX_TABLES = int(os.getenv("ROUTER_MAX_TABLES", "20"))
