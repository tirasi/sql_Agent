import sys
from pathlib import Path

# Add project root to sys.path to resolve imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.nlq_sql_graph import build_graph
from app.state.agent_state import AgentState


def ask(question: str) -> str:
    graph = build_graph()
    initial_state: AgentState = {
        "user_input": question,
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
    final_state = graph.invoke(initial_state)

    print(f"\nQUESTION: {question}")
    print(f"  resolved_input:   {final_state['resolved_input']}")
    print(f"  relevant_tables:  {final_state['relevant_tables']}")
    print(f"  sql_query:        {final_state['sql_query']}")
    print(f"  attempts:         {final_state['attempts']}")
    print(f"  rows:             {final_state['rows']}")
    print(f"  ANSWER:           {final_state['final_answer']}")
    return final_state["final_answer"]


if __name__ == "__main__":
    ask("Which customers in Jaipur placed orders, and what did they buy?")
    ask("How many customers ordered notebooks?")
