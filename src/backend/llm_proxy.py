"""
OpenRouter LLM proxy — keeps API keys on the server in production (R3-05).

Browser calls CardioGuard `/api/llm/*`; this module forwards to OpenRouter.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://cardioguard-ai.local")
DEFAULT_TITLE = "CardioGuard-AI"

FREE_MODEL_CHAIN: List[str] = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


def allow_client_llm_key() -> bool:
    """Dev fallback: accept X-OpenRouter-Key from browser when server key unset."""
    return os.getenv("ALLOW_CLIENT_LLM_KEY", "1") == "1"


def server_key_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def llm_proxy_enabled() -> bool:
    return os.getenv("LLM_PROXY_ENABLED", "1") == "1"


def llm_available() -> bool:
    if not llm_proxy_enabled():
        return False
    return server_key_configured() or allow_client_llm_key()


def resolve_api_key(client_key: Optional[str]) -> str:
    server = os.getenv("OPENROUTER_API_KEY", "").strip()
    if server:
        return server
    if allow_client_llm_key() and client_key and client_key.strip():
        return client_key.strip()
    raise HTTPException(
        status_code=401,
        detail=(
            "LLM proxy: set OPENROUTER_API_KEY on the server "
            "or enable ALLOW_CLIENT_LLM_KEY for local dev with X-OpenRouter-Key"
        ),
    )


def openrouter_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": DEFAULT_REFERER,
        "X-Title": DEFAULT_TITLE,
    }


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
        return response.text[:300]
    except Exception:
        return response.text[:300] or response.reason_phrase


async def proxy_chat_completion(
    body: Dict[str, Any],
    client_key: Optional[str],
) -> Response | StreamingResponse:
    """Forward chat completion to OpenRouter (sync JSON or SSE stream)."""
    if not llm_proxy_enabled():
        raise HTTPException(status_code=503, detail="LLM proxy disabled")

    api_key = resolve_api_key(client_key)
    headers = openrouter_headers(api_key)
    stream = bool(body.get("stream", False))
    timeout = httpx.Timeout(90.0, connect=15.0)

    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        request = client.build_request(
            "POST",
            OPENROUTER_CHAT_URL,
            json=body,
            headers=headers,
        )
        response = await client.send(request, stream=True)

        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            await client.aclose()
            detail = error_body.decode("utf-8", errors="replace")[:300]
            raise HTTPException(status_code=response.status_code, detail=detail)

        async def event_stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            event_stream(),
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.post(
            OPENROUTER_CHAT_URL,
            json=body,
            headers=headers,
        )
        if upstream.status_code >= 400:
            raise HTTPException(
                status_code=upstream.status_code,
                detail=_error_detail(upstream),
            )
        return Response(
            content=upstream.content,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
