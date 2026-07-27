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