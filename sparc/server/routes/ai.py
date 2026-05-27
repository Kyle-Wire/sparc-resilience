"""AI proxy routes — Anthropic key stored in OS keychain.

Key is never transmitted to or held by the frontend.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["ai"])

_AI_KEYRING_SERVICE = "SPARC Labs"
_AI_KEYRING_USERNAME = "anthropic_api_key"


def _keyring_get() -> str | None:
    """Return the stored Anthropic key, or None if keyring is unavailable / empty."""
    try:
        import keyring
        return keyring.get_password(_AI_KEYRING_SERVICE, _AI_KEYRING_USERNAME) or None
    except Exception:
        return None


def _keyring_set(key: str) -> None:
    import keyring
    keyring.set_password(_AI_KEYRING_SERVICE, _AI_KEYRING_USERNAME, key)


def _keyring_delete() -> None:
    try:
        import keyring
        keyring.delete_password(_AI_KEYRING_SERVICE, _AI_KEYRING_USERNAME)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class _AiChatMessage(BaseModel):
    role: str
    content: str

    model_config = {"extra": "ignore"}


class _AiChatRequest(BaseModel):
    messages: list[_AiChatMessage]
    system: str = ""
    max_tokens: int = Field(default=1024, ge=1, le=8192)

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ai/key")
async def ai_key_status() -> dict[str, Any]:
    """Return whether an Anthropic API key is stored (never returns the key itself)."""
    stored = _keyring_get()
    return {"configured": stored is not None}


@router.put("/ai/key")
async def ai_key_save(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Store an Anthropic API key in the OS keychain.

    Body: ``{ "key": "sk-ant-..." }``
    """
    key = body.get("key", "").strip()
    if not key:
        raise HTTPException(400, "key is required")
    if not key.startswith("sk-ant-"):
        raise HTTPException(400, "key must start with sk-ant-")
    try:
        _keyring_set(key)
    except Exception as exc:
        raise HTTPException(500, f"Failed to store key in OS keychain: {exc}")
    return {"status": "saved"}


@router.delete("/ai/key")
async def ai_key_delete() -> dict[str, Any]:
    """Remove the stored Anthropic API key from the OS keychain."""
    _keyring_delete()
    return {"status": "deleted"}


@router.post("/ai/chat")
async def ai_chat(req: _AiChatRequest) -> Any:
    """Proxy a chat turn to the Anthropic Claude API using the key from the OS keychain.

    Returns 401 if no key is configured so the UI can prompt the user to add one.
    """
    import json as _json
    import urllib.error
    import urllib.request

    api_key = _keyring_get()
    if not api_key:
        raise HTTPException(401, "No Anthropic API key configured. Add one in Settings.")

    payload = {
        "model": "claude-sonnet-4-6-20250514",
        "max_tokens": req.max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
    }
    if req.system:
        payload["system"] = req.system

    data = _json.dumps(payload).encode()
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise HTTPException(exc.code, f"Anthropic API error: {body}")
    except Exception as exc:
        raise HTTPException(502, f"Failed to reach Anthropic API: {exc}")
