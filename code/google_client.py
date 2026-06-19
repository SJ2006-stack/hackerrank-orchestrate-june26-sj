"""Google Gemini API client for vision + JSON completions."""

from __future__ import annotations

import base64
import os
import time
from typing import Any

RETRYABLE_HINTS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "quota",
    "rate limit",
    "overloaded",
    "timeout",
    "deadline",
    "unavailable",
)


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
    return key


def _request_timeout_s() -> float:
    return float(os.environ.get("REQUEST_TIMEOUT_S", "90"))


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(token in message for token in RETRYABLE_HINTS)


def _decode_data_url(url: str) -> tuple[bytes, str]:
    if not url.startswith("data:"):
        raise ValueError("Expected data URL for inline image")
    header, encoded = url.split(",", 1)
    mime = header.split(";")[0].removeprefix("data:")
    if not mime:
        mime = "image/jpeg"
    return base64.b64decode(encoded), mime


def _build_parts(user_content: list[dict[str, Any]]) -> list[Any]:
    from google.genai import types

    parts: list[Any] = []
    for block in user_content:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(types.Part.from_text(text=str(block.get("text", ""))))
        elif block_type == "image_url":
            image_url = block.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            data, mime = _decode_data_url(url)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        else:
            raise ValueError(f"Unsupported content block type: {block_type!r}")
    return parts


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion = int(getattr(usage, "candidates_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def google_vision_json_completion(
    model_id: str,
    system: str,
    user_content: list[dict[str, Any]],
    temperature: float = 0,
) -> tuple[str, dict[str, float | int]]:
    """Call a Google vision model and return (content, per-call stats)."""
    from response_cache import get_cached, set_cached

    cached = get_cached("google", model_id, system, user_content)
    if cached is not None:
        return cached["content"], {
            "provider": "google",
            "model_used": model_id,
            "model_calls": 0,
            "retries": 0,
            "total_latency_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit": True,
        }

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_api_key())
    parts = _build_parts(user_content)
    timeout_ms = int(_request_timeout_s() * 1000)

    stats: dict[str, float | int] = {
        "provider": "google",
        "model_used": model_id,
        "model_calls": 0,
        "retries": 0,
        "total_latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        http_options=types.HttpOptions(timeout=timeout_ms),
    )

    last_exc: Exception | None = None
    max_retries = 3
    for attempt in range(max_retries):
        if attempt > 0:
            stats["retries"] = int(stats["retries"]) + 1
            time.sleep(min(2**attempt, 30))

        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )
            elapsed = time.perf_counter() - start
            stats["total_latency_s"] = float(stats["total_latency_s"]) + elapsed
            stats["model_calls"] = int(stats["model_calls"]) + 1

            content = getattr(response, "text", None) or ""
            if not content.strip():
                raise RuntimeError("Empty response content from Google API")

            usage = _usage_from_response(response)
            stats["prompt_tokens"] = usage["prompt_tokens"]
            stats["completion_tokens"] = usage["completion_tokens"]
            stats["total_tokens"] = usage["total_tokens"]
            stats["cache_hit"] = False
            set_cached(
                "google",
                model_id,
                system,
                user_content,
                content,
                usage,
            )
            return content, stats
        except Exception as exc:  # noqa: BLE001
            stats["total_latency_s"] = float(stats["total_latency_s"]) + (
                time.perf_counter() - start
            )
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries - 1:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("google_vision_json_completion failed without an exception")
