"""
Node 1: context_resolution
Normalizes casual phrasing ("notebooks" -> "Notebook") using the synonym map,
BEFORE table selection / SQL generation see the question.

FIX (old version):
- checked case-INsensitively but replaced case-sensitively, so "Notebooks"
  was "detected" yet never replaced;
- matched inside other words: "pens" inside "exPENSive" turned
  "most expensive order" into "most exPen Setive order".
Now: ONE regex, whole words only (\b), case-insensitive, longest term first,
applied in a single pass (so a replacement is never re-replaced).
"""

import re
from app.state.agent_state import AgentState
from app.schema.retriever import get_synonyms


def _build_pattern(terms):
    terms = sorted(terms, key=len, reverse=True)  # "pen set" before "pens"
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def context_resolution_node(state: AgentState) -> dict:
    text = state["user_input"]
    synonyms = get_synonyms()
    if not synonyms:
        return {"resolved_input": text, "detected_synonyms": {}}

    lookup = {k.lower(): v for k, v in synonyms.items()}
    applied = {}

    def _replace(match):
        canonical = lookup[match.group(0).lower()]
        applied[match.group(0)] = canonical
        return canonical

    resolved = _build_pattern(synonyms).sub(_replace, text)
    return {"resolved_input": resolved, "detected_synonyms": applied}
