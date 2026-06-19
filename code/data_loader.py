"""Load claims, user history, and evidence requirements from CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_ALWAYS_APPLIES = frozenset({"general claim review", "reviewability"})
_MULTI_IMAGE_APPLIES = "multi-image rows"

_APPLIES_TO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dent or scratch": (
        "dent",
        "scratch",
        "scrape",
        "bumper",
        "panel",
        "fender",
        "door",
        "hood",
        "trunk",
    ),
    "crack, broken, or missing part": (
        "crack",
        "broken",
        "damaged",
        "missing",
        "shatter",
        "shattered",
    ),
    "vehicle identity or orientation": (
        "identity",
        "orientation",
        "which car",
        "different car",
        "full view",
        "full-view",
    ),
    "screen, keyboard, or trackpad": (
        "screen",
        "keyboard",
        "trackpad",
        "display",
        "lcd",
        "key",
        "keys",
    ),
    "hinge, lid, corner, body, or port": (
        "hinge",
        "lid",
        "corner",
        "body",
        "base",
        "port",
        "usb",
        "charger",
        "casing",
    ),
    "crushed, torn, or seal damage": (
        "crush",
        "crushed",
        "torn",
        "tear",
        "seal",
        "flap",
    ),
    "water, stain, or label damage": (
        "water",
        "stain",
        "wet",
        "label",
        "shipping label",
        "moisture",
    ),
    "contents or inner item": (
        "contents",
        "inside",
        "inner",
        "missing item",
        "not inside",
        "product inside",
    ),
}


def _customer_claim_text(user_claim: str) -> str:
    """Use customer utterances only so support-agent wording does not skew heuristics."""
    customer_parts: list[str] = []
    for segment in user_claim.split("|"):
        segment = segment.strip()
        if segment.lower().startswith("customer:"):
            customer_parts.append(segment.split(":", 1)[1].strip())
    if customer_parts:
        return " ".join(customer_parts).lower()
    return user_claim.lower()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_claims(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def load_user_history(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def parse_image_paths(image_paths: str) -> list[str]:
    return [part.strip() for part in image_paths.split(";") if part.strip()]


def image_path_to_id(image_path: str) -> str:
    return Path(image_path).stem


def resolve_image_path(repo_root: Path, rel_path: str) -> Path:
    """Resolve CSV image path to an on-disk file under the repo."""
    direct = repo_root / rel_path
    if direct.exists():
        return direct
    under_dataset = repo_root / "dataset" / rel_path
    if under_dataset.exists():
        return under_dataset
    return direct


def _claim_matches_applies_to(applies_to: str, claim_lower: str) -> bool:
    keywords = _APPLIES_TO_KEYWORDS.get(applies_to)
    if not keywords:
        return False
    return any(keyword in claim_lower for keyword in keywords)


def relevant_evidence_requirements(
    requirements: list[dict[str, str]],
    claim_object: str,
    user_claim: str = "",
    image_count: int = 1,
) -> list[dict[str, str]]:
    """Return evidence rows relevant to this claim object, text, and image count."""
    claim_lower = _customer_claim_text(user_claim)
    selected: list[dict[str, str]] = []

    for row in requirements:
        if row["claim_object"] not in {"all", claim_object}:
            continue

        applies_to = row["applies_to"]
        if applies_to in _ALWAYS_APPLIES:
            selected.append(row)
            continue
        if applies_to == _MULTI_IMAGE_APPLIES and image_count > 1:
            selected.append(row)
            continue
        if _claim_matches_applies_to(applies_to, claim_lower):
            selected.append(row)

    return selected


def format_user_context(history: dict[str, dict[str, str]], user_id: str) -> dict[str, Any]:
    row = history.get(user_id)
    if not row:
        return {
            "found": False,
            "history_flags": "none",
            "history_summary": "No prior claim history on file.",
        }
    return {
        "found": True,
        "past_claim_count": row.get("past_claim_count", "0"),
        "accept_claim": row.get("accept_claim", "0"),
        "manual_review_claim": row.get("manual_review_claim", "0"),
        "rejected_claim": row.get("rejected_claim", "0"),
        "last_90_days_claim_count": row.get("last_90_days_claim_count", "0"),
        "history_flags": row.get("history_flags", "none"),
        "history_summary": row.get("history_summary", ""),
    }
