"""Normalize and validate model predictions before output."""

from __future__ import annotations

import re
from typing import Any

from schema import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS_BY_CLAIM,
    RISK_FLAGS,
    SEVERITIES,
)

_RISK_FLAG_ORDER = {flag: index for index, flag in enumerate(sorted(RISK_FLAGS))}

_VISUAL_MISMATCH_FLAGS = frozenset(
    {
        "wrong_object",
        "wrong_object_part",
        "claim_mismatch",
        "non_original_image",
    }
)

_INJECTION_PATTERN = re.compile(
    r"(ignore previous|mark as supported|approve this|bypass)", re.IGNORECASE
)


def _normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "true"
    if text in {"false", "no", "0"}:
        return "false"
    return "false"


def _clamp(value: str, allowed: set[str], default: str = "unknown") -> str:
    cleaned = str(value).strip()
    return cleaned if cleaned in allowed else default


def _parse_semicolon_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _normalize_risk_flags(value: Any, *, user_context: dict[str, Any]) -> str:
    flags = _parse_semicolon_list(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for flag in flags:
        if flag in RISK_FLAGS and flag not in seen:
            deduped.append(flag)
            seen.add(flag)

    history_flags = _parse_semicolon_list(user_context.get("history_flags", "none"))
    if "user_history_risk" in history_flags:
        other_risks = [flag for flag in deduped if flag not in {"none", "user_history_risk"}]
        if other_risks and "user_history_risk" not in seen:
            deduped.append("user_history_risk")
            seen.add("user_history_risk")

    if not deduped or deduped == ["none"]:
        return "none"

    deduped = [flag for flag in deduped if flag != "none"]
    if not deduped:
        return "none"

    deduped.sort(key=lambda flag: _RISK_FLAG_ORDER.get(flag, len(_RISK_FLAG_ORDER)))
    return ";".join(deduped)


def _ensure_risk_flag(risk_flags: str, flag: str) -> str:
    flags = _parse_semicolon_list(risk_flags)
    if flag not in flags:
        flags.append(flag)
    flags = [item for item in flags if item != "none"]
    if not flags:
        return "none"
    flags.sort(key=lambda item: _RISK_FLAG_ORDER.get(item, len(_RISK_FLAG_ORDER)))
    return ";".join(flags)


def _normalize_supporting_image_ids(value: Any, image_ids: list[str]) -> str:
    allowed = set(image_ids)
    ids = _parse_semicolon_list(value)
    valid = [image_id for image_id in ids if image_id in allowed]
    if not valid:
        return "none"
    return ";".join(valid)


def _allows_contradicted_without_evidence(risk_flags: str) -> bool:
    return bool(_VISUAL_MISMATCH_FLAGS.intersection(_parse_semicolon_list(risk_flags)))


def apply_postprocess(
    row: dict[str, Any],
    *,
    claim: dict[str, str],
    image_ids: list[str],
    user_context: dict[str, Any],
) -> dict[str, str]:
    """Clamp enums, normalize flags/IDs/booleans, and return a string-valued row."""
    claim_object = claim.get("claim_object", "")
    object_parts = OBJECT_PARTS_BY_CLAIM.get(claim_object, {"unknown"})

    result = dict(row)
    result["evidence_standard_met"] = _normalize_bool(
        result.get("evidence_standard_met", "false")
    )
    result["valid_image"] = _normalize_bool(result.get("valid_image", "false"))
    result["issue_type"] = _clamp(
        str(result.get("issue_type", "unknown")), ISSUE_TYPES
    )
    result["object_part"] = _clamp(
        str(result.get("object_part", "unknown")), object_parts
    )
    result["claim_status"] = _clamp(
        str(result.get("claim_status", "not_enough_information")),
        CLAIM_STATUSES,
        default="not_enough_information",
    )
    result["severity"] = _clamp(str(result.get("severity", "unknown")), SEVERITIES)
    result["risk_flags"] = _normalize_risk_flags(
        result.get("risk_flags"), user_context=user_context
    )
    result["supporting_image_ids"] = _normalize_supporting_image_ids(
        result.get("supporting_image_ids"), image_ids
    )

    history_flags = _parse_semicolon_list(user_context.get("history_flags", "none"))
    if "manual_review_required" in history_flags:
        result["risk_flags"] = _ensure_risk_flag(
            result["risk_flags"], "manual_review_required"
        )

    if "non_original_image" in _parse_semicolon_list(result["risk_flags"]):
        result["valid_image"] = "false"

    if result["evidence_standard_met"] == "false":
        if not (
            result["claim_status"] == "contradicted"
            and _allows_contradicted_without_evidence(result["risk_flags"])
        ):
            result["claim_status"] = "not_enough_information"

    if (
        result["claim_status"] == "supported"
        and result["supporting_image_ids"] == "none"
    ):
        result["claim_status"] = "not_enough_information"

    if result["issue_type"] == "none":
        result["severity"] = "none"
    elif result["issue_type"] == "unknown" and result["severity"] != "unknown":
        result["severity"] = "unknown"

    if (
        result["claim_status"] == "contradicted"
        and result["supporting_image_ids"] != "none"
    ):
        result["supporting_image_ids"] = "none"

    user_claim = claim.get("user_claim", "")
    if _INJECTION_PATTERN.search(user_claim):
        current_flags = _parse_semicolon_list(result["risk_flags"])
        if "text_instruction_present" not in current_flags:
            result["risk_flags"] = _ensure_risk_flag(
                result["risk_flags"], "text_instruction_present"
            )

    return {key: str(value) for key, value in result.items()}
