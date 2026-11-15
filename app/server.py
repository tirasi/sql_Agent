import sys
from pathlib import Path
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.nlq_sql_graph import build_graph
from app.state.agent_state import AgentState, make_initial_state, make_turn
from app.config import SCHEMA_DATA_DIR, DEBUG_TRACE, HISTORY_TURNS
from app.database.connection import is_column_readable
import logging
import re
import threading
from collections import OrderedDict, deque

log = logging.getLogger("nlq_sql")

# C11: compile the LangGraph ONCE at startup (it used to be rebuilt on every
# request). A compiled graph is stateless, so all requests can share it.
GRAPH = build_graph()

# C8: conversation memory, per browser session, in memory. Bounded: at most
# HISTORY_TURNS turns per session and _MAX_SESSIONS sessions (oldest dropped).
_MAX_SESSIONS = 500
_SESSIONS: "OrderedDict[str, deque]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _get_history(session_id: str | None) -> list:
    if not session_id:
        return []
    with _SESSIONS_LOCK:
        turns = _SESSIONS.get(session_id)
        if turns is not None:
            _SESSIONS.move_to_end(session_id)
        return list(turns or [])


def _save_turn(session_id: str | None, final_state: dict) -> None:
    if not session_id:
        return
    with _SESSIONS_LOCK:
        turns = _SESSIONS.setdefault(session_id, deque(maxlen=HISTORY_TURNS))
        turns.append(make_turn(final_state))
        _SESSIONS.move_to_end(session_id)
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)

# S7: fields that can leak schema details or raw DB errors. Sent to the
# browser only when DEBUG_TRACE=true; always logged server-side.
_DEBUG_ONLY_FIELDS = {"validation_error", "execution_error", "attempt_history",
                      "schema_context", "join_context", "table_selection",
                      "candidates", "history"}


def _public_state(node_state: dict) -> dict:
    if DEBUG_TRACE:
        return node_state
    out = {k: v for k, v in node_state.items() if k not in _DEBUG_ONLY_FIELDS}
    for k in ("validation_error", "execution_error"):
        if node_state.get(k):
            out[k] = "hidden (set DEBUG_TRACE=true to see details)"
    return out

app = FastAPI(title="NLQ SQL Agent Dashboard")

class QueryPayload(BaseModel):
    question: str
    session_id: str | None = None   # C8: same id = same conversation

def load_schema_file(filename: str):
    path = os.path.join(SCHEMA_DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/schema")
async def get_schema_info():
    try:
        tables = load_schema_file("tables.json") or {}
        # S3: don't advertise hidden columns in the schema panel either
        tables = {t: {**m, "columns": {c: d for c, d in m.get("columns", {}).items()
                                       if is_column_readable(t, c)}}
                  for t, m in tables.items()}
        rules = load_schema_file("business_rules.json") or []
        synonyms = load_schema_file("synonyms.json") or {}
        relationships = load_schema_file("relationships.json") or []
        return {
            "tables": tables,
            "rules": rules,
            "synonyms": synonyms,
            "relationships": relationships
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_stream(payload: QueryPayload):
    session_id = payload.session_id if payload.session_id and _SESSION_ID.match(payload.session_id) else None
    initial_state = make_initial_state(payload.question, _get_history(session_id))
    
    def event_generator():
        try:
            # Yield started signal
            yield f"data: {json.dumps({'event': 'started'})}\n\n"
            
            final_state = dict(initial_state)
            for step in GRAPH.stream(initial_state, stream_mode="updates"):
                if not step:
                    continue
                node_name = list(step.keys())[0]
                final_state.update(step[node_name] or {})
                node_state = _public_state(step[node_name] or {})
                
                # Filter state to ensure everything is JSON serializable
                serializable_state = {}
                for k, v in node_state.items():
                    try:
                        json.dumps(v)
                        serializable_state[k] = v
                    except (TypeError, OverflowError):
                        serializable_state[k] = str(v)
                
                yield f"data: {json.dumps({'event': 'update', 'node': node_name, 'state': serializable_state})}\n\n"
                
            _save_turn(session_id, final_state)
            yield f"data: {json.dumps({'event': 'completed'})}\n\n"
        except Exception as e:
            log.exception("query failed: %s", payload.question)
            msg = str(e) if DEBUG_TRACE else "Something went wrong while answering. Please try again."
            yield f"data: {json.dumps({'event': 'error', 'error': msg})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount the static directory for CSS/JS
# Note: Mount at "/static" and serve index.html directly on root "/"
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def get_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        # Fallback inline HTML if not created yet
        return HTMLResponse(
            content="<h1>FastAPI Server Running</h1><p>Static files not yet populated.</p>",
            status_code=200
        )
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
