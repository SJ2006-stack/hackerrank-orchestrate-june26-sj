"""Multi-provider LLM clients: Google Gemini API + OpenRouter fallback."""

from __future__ import annotations

import os
import time
from typing import Any

from model_router import ModelRoute

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def default_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _request_timeout_s() -> float:
    return float(os.environ.get("REQUEST_TIMEOUT_S", "90"))


def _build_default_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE") or os.environ.get("X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    from openai import OpenAI

    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    headers = _build_default_headers()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers or None,
        timeout=_request_timeout_s(),
    )


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("429", "500", "502", "503", "504", "rate limit", "overloaded")
    )


def _is_json_mode_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "response_format",
            "json_object",
            "json mode",
            "unsupported",
            "not supported",
        )
    )


def _usage_from_openrouter(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _empty_routed_stats() -> dict[str, Any]:
    return {
        "provider": "",
        "model_used": "",
        "model_tier": 0,
        "models_tried": [],
        "fallback_attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_s": 0.0,
        "model_calls": 0,
        "retries": 0,
        "total_latency_s": 0.0,
        "cache_hit": False,
    }


def _merge_call_stats(accum: dict[str, Any], call_stats: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "model_calls", "retries"):
        accum[key] = int(accum.get(key, 0)) + int(call_stats.get(key, 0))
    accum["total_latency_s"] = float(accum.get("total_latency_s", 0.0)) + float(
        call_stats.get("total_latency_s", 0.0)
    )


def vision_json_completion(
    model: str,
    system: str,
    user_content: list[dict[str, Any]],
    temperature: float = 0,
) -> tuple[str, dict[str, float | int]]:
    """Call an OpenRouter vision model and return (content, per-call stats)."""
    from response_cache import get_cached, set_cached

    cached = get_cached("openrouter", model, system, user_content)
    if cached is not None:
        return cached["content"], {
            "provider": "openrouter",
            "model_used": model,
            "model_calls": 0,
            "retries": 0,
            "total_latency_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit": True,
        }

    client = _get_client()
    stats: dict[str, float | int] = {
        "provider": "openrouter",
        "model_used": model,
        "model_calls": 0,
        "retries": 0,
        "total_latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            stats["retries"] = int(stats["retries"]) + 1
            time.sleep(min(2**attempt, 30))

        retry_attempt = False
        for use_json_mode in (True, False):
            kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            start = time.perf_counter()
            try:
                response = client.chat.completions.create(**kwargs)
                stats["total_latency_s"] = float(stats["total_latency_s"]) + (
                    time.perf_counter() - start
                )
                stats["model_calls"] = int(stats["model_calls"]) + 1
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise RuntimeError("Empty response content from OpenRouter")
                usage = _usage_from_openrouter(response)
                stats["prompt_tokens"] = usage["prompt_tokens"]
                stats["completion_tokens"] = usage["completion_tokens"]
                stats["total_tokens"] = usage["total_tokens"]
                stats["cache_hit"] = False
                set_cached(
                    "openrouter",
                    model,
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
                if use_json_mode and _is_json_mode_unsupported(exc):
                    continue
                if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                    retry_attempt = True
                    break
                raise

        if not retry_attempt:
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("vision_json_completion failed without an exception")


def _dispatch_route(
    route: ModelRoute,
    system: str,
    user_content: list[dict[str, Any]],
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    if route.provider == "google":
        from google_client import google_vision_json_completion

        content, stats = google_vision_json_completion(
            model_id=route.model_id,
            system=system,
            user_content=user_content,
            temperature=temperature,
        )
        stats = dict(stats)
        stats["model_tier"] = route.tier
        return content, stats

    if route.provider == "openrouter":
        content, stats = vision_json_completion(
            model=route.model_id,
            system=system,
            user_content=user_content,
            temperature=temperature,
        )
        stats = dict(stats)
        stats["model_tier"] = route.tier
        return content, stats

    raise ValueError(f"Unknown provider: {route.provider!r}")


def vision_json_completion_routed(
    routes: list[ModelRoute],
    system: str,
    user_content: list[dict[str, Any]],
    temperature: float = 0,
) -> tuple[str, dict[str, Any]]:
    """Try routes in order; return content and unified per-claim stats."""
    if not routes:
        raise ValueError("Model route list is empty")

    from response_cache import get_cached

    stats = _empty_routed_stats()
    models_tried: list[dict[str, str]] = []
    last_error: str | None = None

    for index, route in enumerate(routes):
        if index > 0:
            stats["fallback_attempts"] = int(stats["fallback_attempts"]) + 1

        cached = get_cached(route.provider, route.model_id, system, user_content)
        if cached is not None:
            stats["provider"] = route.provider
            stats["model_used"] = route.model_id
            stats["model_tier"] = route.tier
            stats["cache_hit"] = True
            stats["models_tried"] = models_tried
            return cached["content"], stats

        try:
            content, call_stats = _dispatch_route(
                route, system, user_content, temperature
            )
            _merge_call_stats(stats, call_stats)
            stats["provider"] = str(call_stats.get("provider", route.provider))
            stats["model_used"] = str(call_stats.get("model_used", route.model_id))
            stats["model_tier"] = int(call_stats.get("model_tier", route.tier))
            stats["cache_hit"] = bool(call_stats.get("cache_hit", False))
            stats["latency_s"] = float(stats["total_latency_s"])
            stats["models_tried"] = models_tried
            return content, stats
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            partial = getattr(exc, "partial_stats", None)
            if isinstance(partial, dict):
                _merge_call_stats(stats, partial)
            models_tried.append(
                {
                    "provider": route.provider,
                    "model": route.model_id,
                    "error": error_text,
                }
            )
            last_error = error_text
            continue

    stats["models_tried"] = models_tried
    stats["latency_s"] = float(stats["total_latency_s"])
    err = RuntimeError(
        f"All {len(routes)} model routes failed. Last error: {last_error}"
    )
    err.partial_stats = stats  # type: ignore[attr-defined]
    raise err
