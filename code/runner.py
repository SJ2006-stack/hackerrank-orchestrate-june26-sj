"""Shared claim-processing and evaluation runners for the CLI."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from data_loader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
    parse_image_paths,
    resolve_image_path,
)
from evaluation.iteration_store import record_iteration
from evaluation.metrics import (
    claim_status_confusion_matrix,
    export_mismatches,
    summarize,
)
from model_router import cascade_label
from output_writer import write_predictions
from verifier import ClaimVerifier

CODE_ROOT = Path(__file__).resolve().parent


def default_repo_root() -> Path:
    return CODE_ROOT.parent


def load_env(repo_root: Path | None = None) -> Path:
    from dotenv import load_dotenv

    root = (repo_root or default_repo_root()).resolve()
    load_dotenv(root / ".env")
    return root


def build_verifier(repo_root: Path, model: str | None = None) -> ClaimVerifier:
    return ClaimVerifier(
        repo_root=repo_root,
        user_history=load_user_history(repo_root / "dataset" / "user_history.csv"),
        evidence_requirements=load_evidence_requirements(
            repo_root / "dataset" / "evidence_requirements.csv"
        ),
        model=model,
    )


def process_claims(
    verifier: ClaimVerifier,
    claims: list[dict[str, str]],
    *,
    verbose: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, str]] = []
    timings: list[dict[str, Any]] = []
    per_request_usage: list[dict[str, Any]] = []
    total = len(claims)

    for index, claim in enumerate(claims, start=1):
        latency_before = float(verifier.stats["total_latency_s"])
        calls_before = int(verifier.stats["model_calls"])
        tokens_before = int(verifier.stats["total_tokens"])
        start = time.perf_counter()
        prediction = verifier.verify_claim(claim)
        wall_s = time.perf_counter() - start
        model_latency_s = float(verifier.stats["total_latency_s"]) - latency_before
        model_calls = int(verifier.stats["model_calls"]) - calls_before
        claim_tokens = int(verifier.stats["total_tokens"]) - tokens_before
        meta = dict(verifier.last_call_meta)

        predictions.append(prediction)
        timing = {
            "index": index,
            "user_id": claim["user_id"],
            "claim_object": claim.get("claim_object", ""),
            "image_count": len(parse_image_paths(claim["image_paths"])),
            "model_calls": model_calls,
            "model_latency_s": round(model_latency_s, 3),
            "latency_s": round(model_latency_s, 3),
            "wall_clock_s": round(wall_s, 3),
            "predicted_claim_status": prediction.get("claim_status", ""),
            "provider": meta.get("provider", ""),
            "model_used": meta.get("model_used", ""),
            "prompt_tokens": int(meta.get("prompt_tokens", 0)),
            "completion_tokens": int(meta.get("completion_tokens", 0)),
            "total_tokens": int(meta.get("total_tokens", claim_tokens)),
            "fallback_attempts": int(meta.get("fallback_attempts", 0)),
        }
        timings.append(timing)
        per_request_usage.append(
            {
                "index": index,
                "user_id": claim["user_id"],
                "provider": timing["provider"],
                "model": timing["model_used"],
                "model_tier": meta.get("model_tier", 0),
                "models_tried": meta.get("models_tried", []),
                "fallback_attempts": timing["fallback_attempts"],
                "prompt_tokens": timing["prompt_tokens"],
                "completion_tokens": timing["completion_tokens"],
                "total_tokens": timing["total_tokens"],
                "latency_s": timing["latency_s"],
            }
        )
        if verbose:
            model_info = timing["model_used"] or "n/a"
            print(
                f"[{index}/{total}] user={claim['user_id']} "
                f"model={model_info} model_latency={model_latency_s:.2f}s "
                f"wall={wall_s:.2f}s tokens={timing['total_tokens']} "
                f"status={prediction.get('claim_status', '')}"
            )

    return predictions, timings, per_request_usage


def print_stats(verifier: ClaimVerifier, row_count: int) -> None:
    total_latency = float(verifier.stats["total_latency_s"])
    avg = total_latency / row_count if row_count else 0.0
    models_used = verifier.stats.get("models_used", [])
    providers_used = verifier.stats.get("providers_used", [])
    print(
        "Stats:",
        f"model={verifier.model_label}",
        f"model_calls={verifier.stats['model_calls']}",
        f"retries={verifier.stats['retries']}",
        f"total_latency_s={total_latency:.2f}",
        f"avg_latency_s_per_claim={avg:.2f}",
        f"images_processed={verifier.stats['images_processed']}",
        f"prompt_tokens={verifier.stats['prompt_tokens']}",
        f"completion_tokens={verifier.stats['completion_tokens']}",
        f"total_tokens={verifier.stats['total_tokens']}",
        f"providers={','.join(providers_used) if providers_used else 'n/a'}",
        f"models={','.join(models_used) if models_used else 'n/a'}",
    )


def run_predictions(
    *,
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    model: str | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    claims = load_claims(input_path)
    if limit is not None:
        claims = claims[:limit]

    verifier = build_verifier(repo_root, model=model)
    predictions, timings, per_request_usage = process_claims(verifier, claims, verbose=verbose)
    write_predictions(output_path, predictions)

    print(f"Wrote {len(predictions)} rows to {output_path}")
    print_stats(verifier, len(claims))
    return {
        "predictions": predictions,
        "timings": timings,
        "per_request_usage": per_request_usage,
        "verifier": verifier,
        "row_count": len(claims),
    }


def evaluate_sample(
    *,
    repo_root: Path,
    strategy: str,
    model: str | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
    record_iteration_snapshot: bool = False,
    notes: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    sample_path = repo_root / "dataset" / "sample_claims.csv"
    runs_dir = (output_dir or CODE_ROOT / "evaluation" / "runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    labeled_rows = load_claims(sample_path)
    if limit is not None:
        labeled_rows = labeled_rows[:limit]

    input_rows = [
        {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
        }
        for row in labeled_rows
    ]

    verifier = build_verifier(repo_root, model=model)
    predictions, timings, per_request_usage = process_claims(verifier, input_rows, verbose=verbose)

    predictions_path = runs_dir / f"sample_predictions_{strategy}.csv"
    write_predictions(predictions_path, predictions)

    metrics = summarize(labeled_rows, predictions)
    metrics["strategy"] = strategy
    metrics["model"] = verifier.model_label
    metrics["model_calls"] = verifier.stats["model_calls"]
    metrics["images_processed"] = verifier.stats["images_processed"]
    row_count = len(labeled_rows)
    total_latency = float(verifier.stats["total_latency_s"])
    metrics["total_latency_s"] = total_latency
    metrics["avg_latency_s_per_claim"] = total_latency / row_count if row_count else 0.0
    metrics["prompt_tokens"] = verifier.stats["prompt_tokens"]
    metrics["completion_tokens"] = verifier.stats["completion_tokens"]
    metrics["total_tokens"] = verifier.stats["total_tokens"]
    metrics["providers_used"] = list(verifier.stats.get("providers_used", []))
    metrics["models_used"] = list(verifier.stats.get("models_used", []))
    metrics["cache_hits"] = int(verifier.stats.get("cache_hits", 0))

    metrics_path = runs_dir / f"metrics_{strategy}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion = claim_status_confusion_matrix(labeled_rows, predictions)
    confusion_path = runs_dir / f"claim_status_confusion_{strategy}.json"
    confusion_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    mismatches_path = runs_dir / f"mismatches_{strategy}.csv"
    mismatch_count = export_mismatches(labeled_rows, predictions, mismatches_path)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote confusion matrix to {confusion_path}")
    print(f"Wrote {mismatch_count} mismatched rows to {mismatches_path}")

    iteration_dir: Path | None = None
    if record_iteration_snapshot:
        iteration_dir = record_iteration(
            strategy=strategy,
            model=verifier.model_label,
            metrics=metrics,
            artifact_paths={
                "metrics": metrics_path,
                "predictions": predictions_path,
                "confusion": confusion_path,
                "mismatches": mismatches_path,
            },
            notes=notes,
            per_claim_timings=timings,
            per_request_usage=per_request_usage,
        )
        print(f"Recorded iteration snapshot at {iteration_dir}")

    return {
        "metrics": metrics,
        "mismatch_count": mismatch_count,
        "iteration_dir": iteration_dir,
        "paths": {
            "predictions": predictions_path,
            "metrics": metrics_path,
            "mismatches": mismatches_path,
        },
    }


def check_setup(repo_root: Path) -> int:
    """Validate local setup without calling the API. Returns exit code."""
    ok = True
    print(f"Repo root: {repo_root}")

    env_path = repo_root / ".env"
    if env_path.exists():
        print(f"  [ok] {env_path.relative_to(repo_root)}")
    else:
        print(f"  [!!] Missing {env_path.relative_to(repo_root)} — copy from .env.example")
        ok = False

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if gemini_key:
        masked = gemini_key[:8] + "..." + gemini_key[-4:] if len(gemini_key) > 12 else "[set]"
        print(f"  [ok] GEMINI_API_KEY={masked}")
    else:
        print("  [!!] GEMINI_API_KEY not set (required for Google model cascade)")
        ok = False

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "[set]"
        print(f"  [ok] OPENROUTER_API_KEY={masked} (fallback tier)")
    else:
        print("  [warn] OPENROUTER_API_KEY not set — OpenRouter fallback disabled")

    print(f"  [ok] Model cascade={cascade_label()}")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    print(f"  [ok] OPENROUTER_BASE_URL={base_url}")
    timeout = os.environ.get("REQUEST_TIMEOUT_S", "90")
    print(f"  [ok] REQUEST_TIMEOUT_S={timeout}")

    required_files = [
        "dataset/claims.csv",
        "dataset/sample_claims.csv",
        "dataset/user_history.csv",
        "dataset/evidence_requirements.csv",
    ]
    for rel in required_files:
        path = repo_root / rel
        if path.exists():
            print(f"  [ok] {rel}")
        else:
            print(f"  [!!] Missing {rel}")
            ok = False

    sample = load_claims(repo_root / "dataset" / "sample_claims.csv")
    if sample:
        first_path = parse_image_paths(sample[0]["image_paths"])[0]
        resolved = resolve_image_path(repo_root, first_path)
        if resolved.exists():
            print(f"  [ok] Sample image resolves: {first_path}")
        else:
            print(f"  [!!] Sample image missing: {first_path}")
            ok = False

    if ok:
        print("\nReady. Try: python main.py smoke")
        return 0
    print("\nFix the issues above before running API commands.")
    return 1


def list_iterations() -> int:
    registry_path = CODE_ROOT / "evaluation" / "iterations" / "registry.json"
    if not registry_path.exists():
        print("No iterations recorded yet.")
        print("Run: python main.py evaluate --strategy baseline_v1 --record-iteration")
        return 0

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = registry.get("iterations", [])
    if not rows:
        print("No iterations recorded yet.")
        return 0

    print(f"{'ID':<12} {'Strategy':<20} {'claim_status_acc':>16} {'avg_latency_s':>14}  Recorded")
    print("-" * 80)
    for entry in rows:
        metrics = entry.get("metrics", {})
        acc = metrics.get("claim_status_accuracy")
        acc_s = f"{acc:.1%}" if isinstance(acc, (int, float)) else "n/a"
        lat = metrics.get("avg_latency_s_per_claim")
        lat_s = f"{lat:.2f}" if isinstance(lat, (int, float)) else "n/a"
        print(
            f"{entry.get('iteration_id', '?'):<12} "
            f"{entry.get('strategy', '?'):<20} "
            f"{acc_s:>16} "
            f"{lat_s:>14}  "
            f"{entry.get('recorded_at', '')}"
        )
    print(f"\nFull registry: {registry_path}")
    return 0
