# NLQ-to-SQL Agent

A natural language to SQL agent built with LangGraph that turns plain-English questions into validated, executable SQL queries — with full visibility into every reasoning step along the way.

Ask "Which customers in Jaipur placed orders, and what did they buy?" and get back a correct, schema-aware SQL query, not a guess.

## Overview

Most NLQ-to-SQL demos are a single prompt-and-pray call to an LLM. This project treats it as a proper multi-stage pipeline, because that's what production SQL generation actually requires: understanding what the user meant, knowing what data is available, writing syntactically and semantically correct SQL, and being able to recover when the first attempt is wrong.

The agent is built as a LangGraph state machine with these stages:

1. **Context Resolution** — normalizes business rules, synonyms, and shorthand in the user's question (e.g. "notebooks" → `Notebook`, "pens" → `Pen Set`) into a resolved, unambiguous query.
2. **Schema Retrieval** — identifies the relevant tables and columns for the question, and resolves the join paths between them (foreign key relationships), so the model never has to guess how tables relate.
3. **SQL Generation** — drafts a SQLite query using the resolved input, schema context, and join paths. On validation failure, retries using a dedicated correction prompt that includes the previous query and the specific error, up to a configurable retry limit.
4. **Validation (read-only guard)** — checks the SQL with SQLite's parser on a read-only connection (read-only file + `query_only` + an authorizer allow-list). Any write (DELETE/DROP/UPDATE) is refused, never retried.
5. **Execution** — runs the query; runtime errors are fed back into the retry loop.
6. **Response Generation** — turns the query results back into a plain-language answer. If retries run out, a `give_up` node explains why instead of returning nothing.

Every stage is inspectable through an execution trace UI, so you can see exactly what the agent resolved, which tables it selected, and what SQL it produced — and diagnose failures at the exact step they occur rather than treating the whole pipeline as a black box.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Context Resolution  │  → normalizes synonyms & business rules
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Schema Retrieval     │  → selects target tables + join paths
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  SQL Generation       │  → drafts query (retries on validation failure)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Validation            │  → checks generated SQL before execution
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Response Generation │  → plain-language answer from results
└─────────────────────┘
```

Each node is a discrete, testable unit in the LangGraph state graph, with shared state (`AgentState`) carrying the resolved input, schema context, join context, generated SQL, attempt count, and validation errors between stages.

## Model Routing

Rather than using one model for every step, each node is routed to a model sized for what that step actually requires — classification-style steps use small, fast models; SQL generation (the step that actually determines correctness) uses a coding-tuned model.

| Node | Local (Ollama) | Hosted (Anthropic) |
|---|---|---|
| Schema Retrieval | `phi4-mini` | `claude-haiku-4-5` |
| SQL Generation | `qwen2.5-coder:3b` | `claude-sonnet-4-6` |
| Response Generation | `qwen2.5:3b` | `claude-haiku-4-5` |

Deterministic steps (schema loading, validation, query execution) never call an LLM at all.

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, **or** an Anthropic API key

### Installation

```bash
git clone https://github.com/<your-username>/nlq-sql-agent.git
cd nlq-sql-agent
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and set your provider:

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=ollama          # "ollama" or "anthropic"
OLLAMA_HOST=http://localhost:11434
ANTHROPIC_API_KEY=           # required only if LLM_PROVIDER=anthropic
DB_PATH=dummy.db
MAX_SQL_ATTEMPTS=3
SCHEMA_DATA_DIR=schema_data
```

### Running with Ollama (free, local)

Pull the models used by each node:

```bash
ollama pull phi4-mini
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5:3b
```

Make sure Ollama is running:

```bash
ollama serve
```

If your server is busy on 11434 use this command after pulling models
```bash
for model in gemma3:4b qwen2.5-coder:7b phi4-mini:latest llama3.2:latest; do
  curl http://localhost:11434/api/generate \
    -d "{\"model\":\"$model\",\"prompt\":\"\",\"keep_alive\":-1}" \
    -o /dev/null
done
```

and then to check
```bash
ollama ps
```

Then start the app:

```bash
python app/main.py
```

### Running with Anthropic

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`, then run the same start command. No local models required.

## Project Structure

```
nlq_sql_app/
├── app/
│   ├── graph/              # LangGraph nodes (context_resolution, schema_retrieval, sql_generation, ...)
│   ├── llm/
│   │   └── client.py       # single call_llm() entry point, provider + per-node model routing
│   ├── prompts/            # prompt templates (sql_generation.md, sql_retry.md, ...)
│   ├── state/
│   │   └── agent_state.py  # shared LangGraph state definition
│   ├── config.py            # environment-driven configuration
│   └── main.py
├── schema_data/            # table/column/relationship definitions
├── dummy.db                 # sample SQLite database
├── .env.example
└── requirements.txt
```

## Design Principles

- **Free and self-hostable by default.** The default configuration requires no paid API — everything runs locally through Ollama.
- **No hidden fallbacks.** Every stage either produces a real, schema-grounded result or fails loudly with a clear error, rather than silently returning a generic default.
- **Retry with context, not from scratch.** Failed SQL generation attempts are corrected using the specific validation error and the previous query, not a blind re-roll.
- **Model size matched to task.** Small, fast models handle classification-style steps; a coding-tuned model handles the step that actually determines correctness.
- **Full execution transparency.** Every intermediate step — resolved input, target tables, join paths, generated query, attempt count — is inspectable, not just the final answer.

## Roadmap

- [ ] Support for Postgres/MySQL alongside SQLite
- [ ] Configurable schema auto-discovery from a live database connection
- [ ] Query result caching for repeated questions
- [ ] Expanded synonym/business-rule mapping via a configurable dictionary

## License

MIT
