"""Thin HTTP client for local Ollama embeddings and chat."""

from __future__ import annotations

import httpx

OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
TIMEOUT_SECONDS = 120.0

_FIXIT = (
    "Could not reach Ollama at http://localhost:11434. "
    "Start Ollama, then run:\n"
    "  ollama pull nomic-embed-text\n"
    "  ollama pull llama3.2"
)


class OllamaError(Exception):
    """Raised when Ollama is unreachable or a model is missing."""


def _pull_hint(model: str) -> str:
    return (
        f"Ollama does not have the model '{model}'. Pull it with:\n"
        f"  ollama pull {model}"
    )


def _request(method: str, path: str, payload: dict, model: str) -> dict:
    try:
        response = httpx.request(
            method,
            f"{OLLAMA_HOST}{path}",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.ConnectError as exc:
        raise OllamaError(_FIXIT) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama timed out after {TIMEOUT_SECONDS:.0f}s. "
            "Is a model still downloading, or is the machine under load?"
        ) from exc

    if response.status_code == 404:
        raise OllamaError(_pull_hint(model))

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaError(
            f"Ollama returned a non-JSON response (HTTP {response.status_code})."
        ) from exc

    if response.status_code >= 400:
        detail = data.get("error") if isinstance(data, dict) else None
        if detail and "not found" in str(detail).lower():
            raise OllamaError(_pull_hint(model))
        raise OllamaError(detail or f"Ollama HTTP {response.status_code}")

    return data


def embed(text: str) -> list[float]:
    data = _request(
        "POST",
        "/api/embed",
        {"model": EMBED_MODEL, "input": text},
        EMBED_MODEL,
    )
    embeddings = data.get("embeddings")
    if not embeddings:
        raise OllamaError("Ollama embed response did not include embeddings.")
    return embeddings[0]


def chat(system: str, user: str) -> str:
    data = _request(
        "POST",
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        CHAT_MODEL,
    )
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise OllamaError("Ollama chat response did not include a message.")
    return content
