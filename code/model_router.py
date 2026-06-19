"""Model routing cascade: Google Gemini API first, OpenRouter fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["google", "openrouter"]


@dataclass(frozen=True)
class ModelRoute:
    provider: Provider
    model_id: str
    tier: int
    display_name: str
    quota_hint: dict[str, int] | None = None


DEFAULT_MODEL_CASCADE: list[ModelRoute] = [
    ModelRoute("google", "gemma-4-26b-a4b-it", 1, "Gemma 4 26B A4B"),
    ModelRoute("google", "gemma-4-31b-it", 1, "Gemma 4 31B"),
    ModelRoute(
        "google",
        "gemini-2.5-flash-lite",
        2,
        "Gemini 2.5 Flash Lite",
        {"rpm": 10, "tpm": 250_000, "rpd": 20},
    ),
    ModelRoute(
        "google",
        "gemini-3.1-flash-lite",
        2,
        "Gemini 3.1 Flash Lite",
        {"rpm": 15, "tpm": 250_000, "rpd": 500},
    ),
    ModelRoute(
        "google",
        "gemini-3.5-flash",
        2,
        "Gemini 3.5 Flash",
        {"rpm": 5, "tpm": 250_000, "rpd": 20},
    ),
    ModelRoute("openrouter", "openrouter/free", 3, "OpenRouter Free"),
]

_KNOWN_ROUTES: dict[tuple[Provider, str], ModelRoute] = {
    (route.provider, route.model_id): route for route in DEFAULT_MODEL_CASCADE
}


def _parse_route_spec(spec: str) -> ModelRoute:
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"Invalid route spec (expected provider:model_id): {spec!r}")
    provider_raw, model_id = spec.split(":", 1)
    provider = provider_raw.strip().lower()
    model_id = model_id.strip()
    if provider not in {"google", "openrouter"}:
        raise ValueError(f"Unknown provider in route spec: {provider!r}")
    key = (provider, model_id)  # type: ignore[arg-type]
    if key in _KNOWN_ROUTES:
        return _KNOWN_ROUTES[key]
    tier = 3 if provider == "openrouter" else 0
    return ModelRoute(provider, model_id, tier, model_id)  # type: ignore[arg-type]


def load_model_cascade() -> list[ModelRoute]:
    override = os.environ.get("MODEL_ROUTING_ORDER", "").strip()
    if not override:
        return list(DEFAULT_MODEL_CASCADE)
    routes = [_parse_route_spec(part) for part in override.split(",") if part.strip()]
    if not routes:
        raise ValueError("MODEL_ROUTING_ORDER is set but parsed to an empty cascade")
    return routes


def _infer_provider(model_id: str) -> Provider:
    if model_id.startswith("openrouter/"):
        return "openrouter"
    return "google"


def resolve_active_models(single_model: str | None) -> list[ModelRoute]:
    """Return cascade or a single pinned route when ``single_model`` is set."""
    if not single_model:
        return load_model_cascade()
    provider = _infer_provider(single_model)
    key = (provider, single_model)
    if key in _KNOWN_ROUTES:
        return [_KNOWN_ROUTES[key]]
    tier = 3 if provider == "openrouter" else 0
    return [ModelRoute(provider, single_model, tier, single_model)]


def cascade_label(routes: list[ModelRoute] | None = None) -> str:
    """Human-readable label for metrics / iteration snapshots."""
    active = routes or load_model_cascade()
    if len(active) == 1:
        route = active[0]
        return f"{route.provider}:{route.model_id}"
    return "google_cascade→openrouter/free"
