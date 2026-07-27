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
        return _call_ollama(system_prompt, user_message, model)
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


def _call_ollama(system_prompt: str, user_message: str, model: str) -> str:
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
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running? "
            f"Original error: {e}"
        )

    data = resp.json()
    try:
        return data["message"]["content"]
    except KeyError:
        raise RuntimeError(
            f"Unexpected Ollama response shape for model '{model}': {data}. "
            f"Has `ollama pull {model}` been run?"
        )