"""
Shared state passed between every node in the graph.

Kept JSON-serializable on purpose (no live DB connections or client objects
in here) -- that matters if you later add LangGraph checkpointing/persistence.
"""

import operator
from typing import TypedDict, Optional, List, Dict, Any, Annotated


class AgentState(TypedDict):
    # input
    user_input: str
    history: Optional[List[Dict[str, Any]]]   # C8: previous turns of this conversation


    # context_resolution output
    resolved_input: Optional[str]        # user_input after synonym/business-rule normalization
    detected_synonyms: Optional[Dict[str, str]]
    metric_hints: Optional[List[Dict[str, Any]]]   # C4: business terms -> exact SQL
    is_follow_up: Optional[bool]                    # C8
    chosen_sql: Optional[str]                       # C9: reading the user picked
    chosen_interpretation: Optional[str]

    # schema_retrieval output
    relevant_tables: Optional[List[str]]
    table_selection: Optional[Dict[str, List[str]]]  # table -> reasons it was picked
    schema_context: Optional[str]        # column-level schema text for the SQL prompt
    join_context: Optional[str]          # FK join paths between the selected tables

    # sql_generation output
    sql_query: Optional[str]             # the chosen candidate
    attempts: int
    candidates: Optional[List[Dict[str, Any]]]     # C1: every candidate + its validity/result
    confidence: Optional[float]                    # C1: share of candidates that agreed
    candidate_groups: Optional[int]
    needs_clarification: Optional[bool]            # C9
    clarification_options: Optional[List[Dict[str, Any]]]
    empty_result_retry: Optional[bool]             # C1: regenerate once on an empty result
    empty_rechecked: Optional[bool]

    # validation output
    is_valid: Optional[bool]
    validation_error: Optional[str]
    blocked: Optional[bool]              # refused, no retry (see blocked_reason)
    blocked_reason: Optional[str]        # "write_intent" (S9) | "write" | "restricted" (S3)

    # execution output
    rows: Optional[List[Any]]
    columns: Optional[List[str]]
    execution_error: Optional[str]
    truncated: Optional[bool]            # S2: more rows existed than MAX_RESULT_ROWS

    # response_generation / give_up output
    final_answer: Optional[str]

    # Every attempt (SQL + error) is APPENDED here, never overwritten.
    # Annotated[..., operator.add] is a LangGraph "reducer": when a node
    # returns {"attempt_history": [entry]}, LangGraph does old + [entry]
    # instead of replacing the list.
    attempt_history: Annotated[List[Dict[str, Any]], operator.add]


def make_initial_state(question: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentState:
    """Single place that defines a fresh state (used by main.py, server.py
    and tests/run_eval.py, so adding a field never needs 3 edits).
    history = previous turns of the same conversation (C8), newest last."""
    return {
        "user_input": question,
        "history": history or [],
        "metric_hints": None,
        "is_follow_up": False,
        "chosen_sql": None,
        "chosen_interpretation": None,
        "candidates": None,
        "confidence": None,
        "candidate_groups": None,
        "needs_clarification": False,
        "clarification_options": None,
        "empty_result_retry": False,
        "empty_rechecked": False,
        "resolved_input": None,
        "detected_synonyms": None,
        "relevant_tables": None,
        "table_selection": None,
        "schema_context": None,
        "join_context": None,
        "sql_query": None,
        "attempts": 0,
        "is_valid": None,
        "validation_error": None,
        "blocked": None,
        "blocked_reason": None,
        "rows": None,
        "columns": None,
        "execution_error": None,
        "truncated": None,
        "final_answer": None,
        "attempt_history": [],
    }


def make_turn(final_state: dict) -> dict:
    """C8: what one finished question leaves behind for the next one in the
    same conversation (kept small: no rows, no prompts)."""
    return {
        "question": final_state.get("resolved_input") or final_state.get("user_input"),
        "sql": final_state.get("sql_query"),
        "relevant_tables": final_state.get("relevant_tables") or [],
        "answer": final_state.get("final_answer"),
        "clarification_options": final_state.get("clarification_options")
        if final_state.get("needs_clarification") else None,
    }
