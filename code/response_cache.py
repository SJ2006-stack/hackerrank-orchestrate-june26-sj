"""Disk cache for vision LLM responses keyed by prompt + image content."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_CODE_DIR = Path(__file__).resolve().parent


def cache_enabled() -> bool:
    return os.environ.get("CACHE_ENABLED", "1").strip().lower() in {"1", "true", "yes"}


def cache_dir() -> Path:
    raw = os.environ.get("CACHE_DIR", "code/.cache").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _CODE_DIR.parent / raw
    return path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_image_data_url(url: str) -> bytes:
    if not url.startswith("data:"):
        raise ValueError("Expected data URL for inline image")
    _, encoded = url.split(",", 1)
    return base64.b64decode(encoded)


def extract_cache_inputs(
    system: str, user_content: list[dict[str, Any]]
) -> tuple[str, str, list[str]]:
    """Return (system_prompt_hash, user_text_hash, per_image_sha256[])."""
    text_parts: list[str] = []
    image_shas: list[str] = []
    for block in user_content:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "image_url":
            image_url = block.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            image_shas.append(_sha256_bytes(_decode_image_data_url(url)))
    user_text = "\n".join(text_parts)
    return _sha256_text(system), _sha256_text(user_text), image_shas


def make_cache_key(
    provider: str,
    model_id: str,
    system: str,
    user_content: list[dict[str, Any]],
) -> str:
    system_hash, user_text_hash, image_shas = extract_cache_inputs(system, user_content)
    payload = json.dumps(
        [provider, model_id, system_hash, user_text_hash, image_shas],
        sort_keys=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def get_cached(
    provider: str,
    model_id: str,
    system: str,
    user_content: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not cache_enabled():
        return None
    key = make_cache_key(provider, model_id, system, user_content)
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "content" not in data:
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {"content": str(data["content"]), "usage": usage}


def set_cached(
    provider: str,
    model_id: str,
    system: str,
    user_content: list[dict[str, Any]],
    content: str,
    usage: dict[str, Any],
) -> None:
    if not cache_enabled():
        return
    key = make_cache_key(provider, model_id, system, user_content)
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"content": content, "usage": usage}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
