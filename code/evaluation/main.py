#!/usr/bin/env python3
"""Evaluate the verifier on dataset/sample_claims.csv (delegates to shared runner)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import default_repo_root, evaluate_sample, load_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate claim verification on sample data.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing dataset/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pin a single model (Google model ID or OpenRouter slug); default is full cascade",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="primary",
        help="Label for this run (e.g. baseline_v1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sample claims",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation artifacts",
    )
    parser.add_argument(
        "--record-iteration",
        action="store_true",
        help="Archive this run under evaluation/iterations/ and update registry.json",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes stored with the iteration snapshot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = (args.repo_root or default_repo_root()).resolve()
    load_env(repo_root)
    evaluate_sample(
        repo_root=repo_root,
        strategy=args.strategy,
        model=args.model,
        limit=args.limit,
        output_dir=args.output_dir,
        record_iteration_snapshot=args.record_iteration,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
