"""
Node 0: intent_guard  (S9)
Refuses requests to CHANGE data before any LLM call. Previously "Delete all
orders" was only refused if the model happened to write a DELETE; if it
wrote a SELECT instead, the agent silently answered a different question.

Deterministic on purpose: cheap, predictable, testable. It looks for
  - an imperative write verb at the start ("Delete ...", "Please change ...",
    "Can you remove ...")
  - SQL write syntax anywhere ("drop table", "delete from", "insert into",
    "update x set")
Read questions that merely contain these words ("orders updated in May",
"add up the amounts") are not matched.
"""

import re
from app.state.agent_state import AgentState

_POLITE = r"^\s*(?:please\s+|kindly\s+|(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|need)\s+(?:you\s+)?to\s+)?"
_WRITE_VERBS = (r"delete|drop|remove|truncate|insert|update|change|modify|alter|create|rename|"
                r"erase|wipe|clear|purge|overwrite|edit|replace|reset|set|add(?!\s+up\b)")
_IMPERATIVE = re.compile(_POLITE + r"(?:" + _WRITE_VERBS + r")\b", re.IGNORECASE)
_SQL_WRITE = re.compile(
    r"\b(?:drop|truncate|alter)\s+table\b|\bdelete\s+from\b|\binsert\s+into\b|"
    r"\bupdate\s+\w+\s+set\b|\battach\s+database\b|\bpragma\b",
    re.IGNORECASE,
)

REFUSAL = ("Refused: this agent only answers questions about the data -- it "
           "cannot add, change or delete anything. Try asking a question instead, "
           "e.g. \"How many orders are there?\"")


def is_write_request(question: str) -> bool:
    return bool(_IMPERATIVE.search(question) or _SQL_WRITE.search(question))


def intent_guard_node(state: AgentState) -> dict:
    if is_write_request(state["user_input"]):
        return {"blocked": True, "blocked_reason": "write_intent", "final_answer": REFUSAL}
    return {"blocked": False}
