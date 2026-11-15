"""
Node 1: context_resolution
Prepares the question before table selection / SQL generation see it.

1. Synonyms (C4): only VALUE synonyms ("notebooks" -> "Notebook") and TABLE
   synonyms ("clients" -> "customers") are rewritten -- words whose meaning
   doesn't change. Business terms that mean a CALCULATION ("sales",
   "revenue", "spend") are no longer rewritten (the old map turned "total
   sales" into "total orders", i.e. a SUM into a COUNT). They are looked up
   in metrics.json and their exact SQL is handed to the SQL prompt.
   Rewriting: ONE regex, whole words, case-insensitive, longest term first,
   single pass (a replacement is never re-replaced).
2. Follow-ups (C8): "What about Mumbai?" after a Jaipur question is flagged
   is_follow_up, so later nodes reuse the previous turn's tables and SQL.
3. Clarification answers (C9): if the previous turn asked "which did you
   mean? 1) ... 2) ...", a reply like "2", "the second one" or "by total
   spend" picks that reading, and its already-generated SQL is reused.
"""

import re
from app.state.agent_state import AgentState
from app.schema.retriever import get_synonyms, find_metrics

_FOLLOW_UP = re.compile(
    r"^\s*(?:and|but|also|now|then|ok(?:ay)?|so|what about|how about|same for|same but|only|just|"
    r"which of (?:them|those|these))\b|\b(?:them|those|these|they|their|that one|the same)\b",
    re.IGNORECASE,
)
_ORDINALS = {"first": 0, "second": 1, "third": 2, "1st": 0, "2nd": 1, "3rd": 2}


def _build_pattern(terms):
    terms = sorted(terms, key=len, reverse=True)  # "pen set" before "pens"
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def picked_option(reply: str, options: list[dict]) -> int | None:
    """Which clarification option a reply chooses, or None."""
    r = reply.strip().lower()
    m = re.fullmatch(r"(?:option\s*|no\.?\s*|#\s*)?(\d)\s*[.)]?", r)
    if m and 1 <= int(m.group(1)) <= len(options):
        return int(m.group(1)) - 1
    m = re.fullmatch(r"(?:the\s+)?(first|second|third|1st|2nd|3rd)(?:\s+one)?\.?", r)
    if m and _ORDINALS[m.group(1)] < len(options):
        return _ORDINALS[m.group(1)]
    words = {w for w in re.findall(r"[a-z]+", r) if len(w) > 3}
    if words:
        matches = [i for i, o in enumerate(options)
                   if words <= set(re.findall(r"[a-z]+", (o.get("interpretation") or "").lower()))]
        if len(matches) == 1:
            return matches[0]
    return None


def context_resolution_node(state: AgentState) -> dict:
    text = state["user_input"]
    history = state.get("history") or []
    last = history[-1] if history else None
    out: dict = {}

    if last and last.get("clarification_options"):
        i = picked_option(text, last["clarification_options"])
        if i is not None:
            opt = last["clarification_options"][i]
            text = f"{last['question']} (meaning: {opt['interpretation']})"
            out.update(chosen_sql=opt["sql"], chosen_interpretation=opt["interpretation"])

    synonyms = get_synonyms()
    applied = {}
    if synonyms:
        lookup = {k.lower(): v for k, v in synonyms.items()}

        def _replace(match):
            canonical = lookup[match.group(0).lower()]
            applied[match.group(0)] = canonical
            return canonical

        text = _build_pattern(synonyms).sub(_replace, text)

    is_follow_up = bool(last) and "chosen_sql" not in out and bool(_FOLLOW_UP.search(state["user_input"]))
    return {
        "resolved_input": text,
        "detected_synonyms": applied,
        "metric_hints": find_metrics(text),
        "is_follow_up": is_follow_up,
        **out,
    }
