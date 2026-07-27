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
from app.state.agent_state import AgentState
from app.config import SCHEMA_DATA_DIR

app = FastAPI(title="NLQ SQL Agent Dashboard")

class QueryPayload(BaseModel):
    question: str

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
    graph = build_graph()
    initial_state: AgentState = {
        "user_input": payload.question,
        "resolved_input": None,
        "detected_synonyms": None,
        "relevant_tables": None,
        "schema_context": None,
        "sql_query": None,
        "attempts": 0,
        "is_valid": None,
        "validation_error": None,
        "rows": None,
        "columns": None,
        "execution_error": None,
        "final_answer": None,
    }
    
    def event_generator():
        try:
            # Yield started signal
            yield f"data: {json.dumps({'event': 'started'})}\n\n"
            
            for step in graph.stream(initial_state, stream_mode="updates"):
                if not step:
                    continue
                node_name = list(step.keys())[0]
                node_state = step[node_name]
                
                # Filter state to ensure everything is JSON serializable
                serializable_state = {}
                for k, v in node_state.items():
                    try:
                        json.dumps(v)
                        serializable_state[k] = v
                    except (TypeError, OverflowError):
                        serializable_state[k] = str(v)
                
                yield f"data: {json.dumps({'event': 'update', 'node': node_name, 'state': serializable_state})}\n\n"
                
            yield f"data: {json.dumps({'event': 'completed'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
            
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
