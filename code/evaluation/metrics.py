"""Compare predictions against labeled sample_claims.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from schema import CLAIM_STATUSES, OUTPUT_COLUMNS

PREDICTION_FIELDS = OUTPUT_COLUMNS[4:]
CLAIM_STATUS_LABELS = sorted(CLAIM_STATUSES)

PRIMARY_FIELDS = (
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
    "risk_flags",
    "supporting_image_ids",
)

_SET_FIELDS = frozenset({"risk_flags", "supporting_image_ids"})


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _semicolon_set(value: str) -> frozenset[str]:
    text = value.strip()
    if not text or text.lower() == "none":
        return frozenset()
    return frozenset(part.strip() for part in text.split(";") if part.strip())


def _field_values_match(field: str, expected: str, predicted: str) -> bool:
    if field in _SET_FIELDS:
        return _semicolon_set(expected) == _semicolon_set(predicted)
    return expected.strip() == predicted.strip()


def field_accuracy(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, float]:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, got {len(predicted_rows)}"
        )

    totals = {field: 0 for field in PREDICTION_FIELDS}
    correct = {field: 0 for field in PREDICTION_FIELDS}

    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        for field in PREDICTION_FIELDS:
            totals[field] += 1
            if expected.get(field, "").strip() == predicted.get(field, "").strip():
                correct[field] += 1

    return {field: correct[field] / totals[field] for field in PREDICTION_FIELDS}


def primary_field_accuracy(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, float]:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, got {len(predicted_rows)}"
        )

    totals = {field: 0 for field in PRIMARY_FIELDS}
    correct = {field: 0 for field in PRIMARY_FIELDS}

    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        for field in PRIMARY_FIELDS:
            totals[field] += 1
            if _field_values_match(
                field,
                expected.get(field, ""),
                predicted.get(field, ""),
            ):
                correct[field] += 1

    return {field: correct[field] / totals[field] for field in PRIMARY_FIELDS}


def exact_row_match_rate(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> float:
    if not expected_rows:
        return 0.0
    matches = 0
    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        if all(
            expected.get(field, "").strip() == predicted.get(field, "").strip()
            for field in PREDICTION_FIELDS
        ):
            matches += 1
    return matches / len(expected_rows)


def primary_exact_row_match_rate(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> float:
    if not expected_rows:
        return 0.0
    matches = 0
    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        if all(
            _field_values_match(
                field,
                expected.get(field, ""),
                predicted.get(field, ""),
            )
            for field in PRIMARY_FIELDS
        ):
            matches += 1
    return matches / len(expected_rows)


def claim_status_confusion_matrix(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, object]:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, got {len(predicted_rows)}"
        )

    matrix = {
        expected: {predicted: 0 for predicted in CLAIM_STATUS_LABELS}
        for expected in CLAIM_STATUS_LABELS
    }
    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        actual = expected.get("claim_status", "").strip()
        guess = predicted.get("claim_status", "").strip()
        if actual not in matrix:
            actual = "not_enough_information"
        if guess not in matrix[actual]:
            guess = "not_enough_information"
        matrix[actual][guess] += 1

    return {
        "labels": CLAIM_STATUS_LABELS,
        "matrix": matrix,
        "total": len(expected_rows),
    }


def per_object_type_accuracy(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, dict[str, float]]:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, got {len(predicted_rows)}"
        )

    groups: dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        object_type = expected.get("claim_object", "").strip() or "unknown"
        if object_type not in groups:
            groups[object_type] = ([], [])
        groups[object_type][0].append(expected)
        groups[object_type][1].append(predicted)

    breakdown: dict[str, dict[str, float]] = {}
    for object_type, (expected_group, predicted_group) in sorted(groups.items()):
        per_field = field_accuracy(expected_group, predicted_group)
        breakdown[object_type] = {
            "rows": float(len(expected_group)),
            "exact_row_match_rate": exact_row_match_rate(expected_group, predicted_group),
            "primary_exact_row_match_rate": primary_exact_row_match_rate(
                expected_group, predicted_group
            ),
            "claim_status_accuracy": per_field.get("claim_status", 0.0),
            **per_field,
        }
    return breakdown


def export_mismatches(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    path: Path,
) -> int:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, got {len(predicted_rows)}"
        )

    fieldnames = ["user_id", "claim_object"]
    for field in PREDICTION_FIELDS:
        fieldnames.extend([f"expected_{field}", f"predicted_{field}"])

    mismatch_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
            row_mismatch = any(
                expected.get(field, "").strip() != predicted.get(field, "").strip()
                for field in PREDICTION_FIELDS
            )
            if not row_mismatch:
                continue
            mismatch_count += 1
            row = {
                "user_id": expected.get("user_id", ""),
                "claim_object": expected.get("claim_object", ""),
            }
            for field in PREDICTION_FIELDS:
                row[f"expected_{field}"] = expected.get(field, "")
                row[f"predicted_{field}"] = predicted.get(field, "")
            writer.writerow(row)
    return mismatch_count


def summarize(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, object]:
    per_field = field_accuracy(expected_rows, predicted_rows)
    primary_per_field = primary_field_accuracy(expected_rows, predicted_rows)
    return {
        "rows": len(expected_rows),
        "exact_row_match_rate": exact_row_match_rate(expected_rows, predicted_rows),
        "primary_exact_row_match_rate": primary_exact_row_match_rate(
            expected_rows, predicted_rows
        ),
        "per_field_accuracy": per_field,
        "primary_field_accuracy": primary_per_field,
        "claim_status_accuracy": per_field.get("claim_status", 0.0),
        "claim_status_confusion_matrix": claim_status_confusion_matrix(
            expected_rows, predicted_rows
        ),
        "per_object_type_accuracy": per_object_type_accuracy(expected_rows, predicted_rows),
    }
