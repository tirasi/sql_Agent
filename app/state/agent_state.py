"""
Shared state passed between every node in the graph.

Kept JSON-serializable on purpose (no live DB connections or client objects
in here) -- that matters if you later add LangGraph checkpointing/persistence.
Live resources (DB connection, LLM client) are injected into nodes via
closures/dependency-injection in graph/nlq_sql_graph.py instead.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    # input
    user_input: str

    # context_resolution output
    resolved_input: Optional[str]        # user_input after synonym/business-rule normalization
    detected_synonyms: Optional[Dict[str, str]]

    # schema_retrieval output
    relevant_tables: Optional[List[str]]
    schema_context: Optional[str]        # formatted schema text handed to the SQL-writing prompt

    # sql_generation output
    sql_query: Optional[str]
    attempts: int

    # validation output
    is_valid: Optional[bool]
    validation_error: Optional[str]

    # execution output
    rows: Optional[List[Any]]
    columns: Optional[List[str]]
    execution_error: Optional[str]

    # response_generation output
    final_answer: Optional[str]
