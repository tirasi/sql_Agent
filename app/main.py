import sys
from pathlib import Path

# Add project root to sys.path to resolve imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.nlq_sql_graph import build_graph
from app.state.agent_state import make_initial_state
from app.llm.client import check_provider_ready

GRAPH = build_graph()  # build once, reuse for every question


def ask(question: str) -> str:
    final_state = GRAPH.invoke(make_initial_state(question))

    print(f"\nQUESTION: {question}")
    print(f"  resolved_input:   {final_state['resolved_input']}")
    print(f"  relevant_tables:  {final_state['relevant_tables']}")
    for table, why in (final_state.get("table_selection") or {}).items():
        print(f"      {table:<12} <- {'; '.join(why)}")
    for h in final_state.get("attempt_history") or []:
        status = "OK" if not h["error"] else f"FAILED: {h['error']}"
        print(f"  attempt {h['attempt']} [{h['stage']}] {status}")
        print(f"      {' '.join((h['sql'] or '').split())}")
    print(f"  rows:             {final_state['rows']}")
    print(f"  ANSWER:           {final_state['final_answer']}")
    return final_state["final_answer"]


if __name__ == "__main__":
    problems = check_provider_ready()
    if problems:
        print("Not ready:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    ask("Which customers in Jaipur placed orders, and what did they buy?")
    ask("How many customers ordered notebooks?")
