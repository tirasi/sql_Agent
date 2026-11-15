"""
Single entry point every node uses: call_llm(system, user, node_name).

node_name determines WHICH model handles the call -- see config.py for the
routing table. This is the file that changes if you swap providers or
rebalance which node gets which model size; node files themselves never
hardcode a model name.

Providers:
- "ollama"    -- free, local, per-node routing via NODE_MODELS_OLLAMA.
- "anthropic" -- paid API, per-node routing via NODE_MODELS_ANTHROPIC.
"""

from app.config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    OLLAMA_HOST,
    NODE_MODELS_OLLAMA,
    NODE_MODELS_ANTHROPIC,
)


def call_llm(system_prompt: str, user_message: str, node_name: str = "default") -> str:
    if LLM_PROVIDER == "anthropic":
        model = NODE_MODELS_ANTHROPIC.get(node_name, "claude-sonnet-4-6")
        return _call_anthropic(system_prompt, user_message, model)
    elif LLM_PROVIDER == "ollama":
        model = NODE_MODELS_OLLAMA.get(node_name, "phi4-mini")
        return _call_ollama(system_prompt, user_message, model, node_name)
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' -- must be 'ollama' or 'anthropic'."
        )


def _call_anthropic(system_prompt: str, user_message: str, model: str) -> str:
    import anthropic  # pip install anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in .env -- required for LLM_PROVIDER=anthropic"
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _call_ollama(system_prompt: str, user_message: str, model: str, node_name: str = "default") -> str:
    import requests  # pip install requests

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running? "
            f"Original error: {e}"
        )

    # FIX: raise_for_status() threw away Ollama's error body. On a missing
    # model Ollama answers 404 with {"error": "model '...' not found"} --
    # that body is the useful part, so surface it with node + model + fix.
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        hint = f" Fix: ollama pull {model}" if "not found" in str(detail).lower() else ""
        raise RuntimeError(
            f"[{node_name}] Ollama HTTP {resp.status_code} for model '{model}': {detail}.{hint}"
        )

    data = resp.json()
    try:
        return data["message"]["content"]
    except KeyError:
        raise RuntimeError(
            f"Unexpected Ollama response shape for model '{model}': {data}. "
            f"Has `ollama pull {model}` been run?"
        )


def check_provider_ready() -> list[str]:
    """Preflight: returns a list of problems (empty list = ready).
    Run before main.py / the eval so a missing model fails in 1 clear line
    instead of 23 identical crashes."""
    if LLM_PROVIDER == "anthropic":
        return [] if ANTHROPIC_API_KEY else ["ANTHROPIC_API_KEY not set in .env"]
    if LLM_PROVIDER != "ollama":
        return [f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'"]

    import requests
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json()
    except requests.exceptions.RequestException:
        return [f"Could not reach Ollama at {OLLAMA_HOST} -- run `ollama serve`"]

    installed = {m["name"] for m in tags.get("models", [])}
    norm = lambda n: n if ":" in n else n + ":latest"   # "phi4-mini" == "phi4-mini:latest"
    missing = sorted(m for m in set(NODE_MODELS_OLLAMA.values()) if norm(m) not in installed)
    return [f"Model '{m}' not pulled -- run `ollama pull {m}`" for m in missing]
