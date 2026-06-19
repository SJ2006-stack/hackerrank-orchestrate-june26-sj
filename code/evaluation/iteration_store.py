"""Persist evaluation runs as numbered iterations for systematic comparison."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ITERATIONS_ROOT = Path(__file__).resolve().parent / "iterations"
REGISTRY_PATH = ITERATIONS_ROOT / "registry.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"iterations": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(registry: dict[str, Any]) -> None:
    ITERATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def next_iteration_id() -> str:
    registry = _load_registry()
    count = len(registry.get("iterations", [])) + 1
    return f"iter_{count:03d}"


def _latency_values(per_claim_timings: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for entry in per_claim_timings:
        if "latency_s" in entry:
            values.append(float(entry["latency_s"]))
        elif "model_latency_s" in entry:
            values.append(float(entry["model_latency_s"]))
    return values


def record_iteration(
    *,
    strategy: str,
    model: str,
    metrics: dict[str, Any],
    artifact_paths: dict[str, Path],
    notes: str = "",
    per_claim_timings: list[dict[str, Any]] | None = None,
    per_request_usage: list[dict[str, Any]] | None = None,
) -> Path:
    """Copy artifacts into iterations/<id>_<strategy>/ and append registry entry."""
    iteration_id = next_iteration_id()
    folder_name = f"{iteration_id}_{strategy}"
    dest = ITERATIONS_ROOT / folder_name
    dest.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    for name, src in artifact_paths.items():
        if not src.exists():
            continue
        target = dest / src.name
        shutil.copy2(src, target)
        copied[name] = str(target.relative_to(ITERATIONS_ROOT.parent))

    summary_metrics: dict[str, Any] = {
        "rows": metrics.get("rows"),
        "claim_status_accuracy": metrics.get("claim_status_accuracy"),
        "exact_row_match_rate": metrics.get("exact_row_match_rate"),
        "model_calls": metrics.get("model_calls"),
        "images_processed": metrics.get("images_processed"),
        "total_latency_s": metrics.get("total_latency_s"),
        "avg_latency_s_per_claim": metrics.get("avg_latency_s_per_claim"),
        "per_field_accuracy": metrics.get("per_field_accuracy"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "total_tokens": metrics.get("total_tokens"),
        "providers_used": metrics.get("providers_used"),
        "models_used": metrics.get("models_used"),
    }

    if per_request_usage:
        usage_path = dest / "per_request_usage.json"
        usage_path.write_text(json.dumps(per_request_usage, indent=2), encoding="utf-8")
        copied["per_request_usage"] = str(usage_path.relative_to(ITERATIONS_ROOT.parent))
        summary_metrics["total_fallback_attempts"] = sum(
            int(u.get("fallback_attempts", 0)) for u in per_request_usage
        )
        providers = sorted({str(u.get("provider", "")) for u in per_request_usage if u.get("provider")})
        models = sorted({str(u.get("model", "")) for u in per_request_usage if u.get("model")})
        if providers:
            summary_metrics["providers_used"] = providers
        if models:
            summary_metrics["models_used"] = models

    summary = {
        "iteration_id": iteration_id,
        "strategy": strategy,
        "model": model,
        "recorded_at": _utc_now_iso(),
        "notes": notes,
        "metrics": summary_metrics,
        "artifacts": copied,
    }
    if per_claim_timings:
        timings_path = dest / "per_claim_timings.json"
        timings_path.write_text(json.dumps(per_claim_timings, indent=2), encoding="utf-8")
        summary["artifacts"]["per_claim_timings"] = str(
            timings_path.relative_to(ITERATIONS_ROOT.parent)
        )
        latencies = _latency_values(per_claim_timings)
        if latencies:
            summary["metrics"]["min_latency_s_per_claim"] = min(latencies)
            summary["metrics"]["max_latency_s_per_claim"] = max(latencies)
            sorted_lat = sorted(latencies)
            summary["metrics"]["median_latency_s_per_claim"] = sorted_lat[len(sorted_lat) // 2]

    summary_path = dest / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    registry = _load_registry()
    registry.setdefault("iterations", []).append(summary)
    _save_registry(registry)
    return dest
