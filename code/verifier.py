"""Multimodal claim verification with Google-first model routing."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from data_loader import (
    format_user_context,
    image_path_to_id,
    parse_image_paths,
    relevant_evidence_requirements,
    resolve_image_path,
)
from llm_client import vision_json_completion_routed
from model_router import cascade_label, resolve_active_models
from postprocess import apply_postprocess
from prompts import SYSTEM_PROMPT, build_verification_prompt
from schema import OUTPUT_COLUMNS

DEFAULT_MODEL = cascade_label()


def _resize_max_px() -> int | None:
    raw = os.environ.get("RESIZE_MAX_PX", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _read_image_bytes(path: Path) -> bytes:
    max_px = _resize_max_px()
    data = path.read_bytes()
    if max_px is None:
        return data

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    width, height = img.size
    if max(width, height) <= max_px:
        return data

    scale = max_px / max(width, height)
    resized = img.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.LANCZOS,
    )
    buffer = io.BytesIO()
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        resized.save(buffer, format="JPEG", quality=85)
    else:
        fmt = (img.format or "PNG").upper()
        resized.save(buffer, format=fmt)
    return buffer.getvalue()


def _encode_image(path: Path) -> str:
    data = _read_image_bytes(path)
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "true"
    if text in {"false", "no", "0"}:
        return "false"
    return "false"


def _normalize_risk_flags(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        flags = [str(item).strip() for item in value if str(item).strip()]
        return ";".join(flags) if flags else "none"
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "none"
    return text


def _normalize_supporting_ids(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        ids = [str(item).strip() for item in value if str(item).strip()]
        return ";".join(ids) if ids else "none"
    text = str(value).strip()
    return text if text else "none"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    return _coerce_prediction_dict(parsed)


def _coerce_prediction_dict(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
    raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")


def _fallback_prediction(claim: dict[str, str], reason: str) -> dict[str, str]:
    return {
        "user_id": claim["user_id"],
        "image_paths": claim["image_paths"],
        "user_claim": claim["user_claim"],
        "claim_object": claim["claim_object"],
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


class ClaimVerifier:
    def __init__(
        self,
        repo_root: Path,
        user_history: dict[str, dict[str, str]],
        evidence_requirements: list[dict[str, str]],
        model: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.user_history = user_history
        self.evidence_requirements = evidence_requirements
        self.model = model
        self.routes = resolve_active_models(model)
        self.last_call_meta: dict[str, Any] = {}
        self.stats: dict[str, float | int | list[str]] = {
            "model_calls": 0,
            "retries": 0,
            "total_latency_s": 0.0,
            "images_processed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "models_used": [],
            "providers_used": [],
        }

    @property
    def model_label(self) -> str:
        return cascade_label(self.routes) if not self.model else f"pinned:{self.model}"

    def _accumulate_call_stats(self, call_stats: dict[str, Any]) -> None:
        self.stats["model_calls"] = int(self.stats["model_calls"]) + int(
            call_stats.get("model_calls", 0)
        )
        self.stats["retries"] = int(self.stats["retries"]) + int(call_stats.get("retries", 0))
        self.stats["total_latency_s"] = float(self.stats["total_latency_s"]) + float(
            call_stats.get("total_latency_s", 0.0)
        )
        self.stats["prompt_tokens"] = int(self.stats["prompt_tokens"]) + int(
            call_stats.get("prompt_tokens", 0)
        )
        self.stats["completion_tokens"] = int(self.stats["completion_tokens"]) + int(
            call_stats.get("completion_tokens", 0)
        )
        self.stats["total_tokens"] = int(self.stats["total_tokens"]) + int(
            call_stats.get("total_tokens", 0)
        )
        if call_stats.get("cache_hit"):
            self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
        model_used = str(call_stats.get("model_used", ""))
        provider = str(call_stats.get("provider", ""))
        models_used = list(self.stats["models_used"])
        providers_used = list(self.stats["providers_used"])
        if model_used and model_used not in models_used:
            models_used.append(model_used)
        if provider and provider not in providers_used:
            providers_used.append(provider)
        self.stats["models_used"] = models_used
        self.stats["providers_used"] = providers_used

    def verify_claim(self, claim: dict[str, str]) -> dict[str, str]:
        image_paths = parse_image_paths(claim["image_paths"])
        image_ids = [image_path_to_id(path) for path in image_paths]
        user_context = format_user_context(self.user_history, claim["user_id"])
        requirements = relevant_evidence_requirements(
            self.evidence_requirements,
            claim["claim_object"],
            user_claim=claim["user_claim"],
            image_count=len(image_paths),
        )
        prompt = build_verification_prompt(claim, user_context, requirements, image_ids)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for rel_path in image_paths:
            abs_path = resolve_image_path(self.repo_root, rel_path)
            if not abs_path.exists():
                self.last_call_meta = {"error": f"Missing image file: {rel_path}"}
                return _fallback_prediction(
                    claim, f"Missing image file: {rel_path}"
                )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(abs_path)},
                }
            )
            self.stats["images_processed"] = int(self.stats["images_processed"]) + 1

        try:
            raw_content, call_stats = vision_json_completion_routed(
                routes=self.routes,
                system=SYSTEM_PROMPT,
                user_content=content,
                temperature=0,
            )
            self._accumulate_call_stats(call_stats)
            self.last_call_meta = dict(call_stats)
            raw_response = raw_content
        except Exception as exc:  # noqa: BLE001 - surface model failures as review rows
            partial = getattr(exc, "partial_stats", None)
            if isinstance(partial, dict):
                self._accumulate_call_stats(partial)
                self.last_call_meta = dict(partial)
                self.last_call_meta["error"] = str(exc)
            else:
                self.last_call_meta = {"error": str(exc)}
            return _fallback_prediction(claim, f"Model call failed: {exc}")

        try:
            parsed = _extract_json(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            self.last_call_meta["parse_error"] = str(exc)
            return _fallback_prediction(claim, f"Invalid model JSON: {exc}")

        row = self._to_output_row(claim, parsed)
        return apply_postprocess(
            row, claim=claim, image_ids=image_ids, user_context=user_context
        )

    def _to_output_row(self, claim: dict[str, str], parsed: dict[str, Any]) -> dict[str, str]:
        row = {
            "user_id": claim["user_id"],
            "image_paths": claim["image_paths"],
            "user_claim": claim["user_claim"],
            "claim_object": claim["claim_object"],
            "evidence_standard_met": _normalize_bool(parsed.get("evidence_standard_met")),
            "evidence_standard_met_reason": str(
                parsed.get("evidence_standard_met_reason", "")
            ).strip(),
            "risk_flags": _normalize_risk_flags(parsed.get("risk_flags")),
            "issue_type": str(parsed.get("issue_type", "unknown")).strip(),
            "object_part": str(parsed.get("object_part", "unknown")).strip(),
            "claim_status": str(parsed.get("claim_status", "not_enough_information")).strip(),
            "claim_status_justification": str(
                parsed.get("claim_status_justification", "")
            ).strip(),
            "supporting_image_ids": _normalize_supporting_ids(
                parsed.get("supporting_image_ids")
            ),
            "valid_image": _normalize_bool(parsed.get("valid_image")),
            "severity": str(parsed.get("severity", "unknown")).strip(),
        }
        for column in OUTPUT_COLUMNS:
            row.setdefault(column, "")
        return row
