#!/usr/bin/env python3
"""CLI for damage-claim evidence verification (HackerRank Orchestrate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runner import (
    check_setup,
    default_repo_root,
    evaluate_sample,
    list_iterations,
    load_env,
    run_predictions,
)

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing dataset/ (default: parent of code/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pin a single model (Google model ID or OpenRouter slug); default is full cascade",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrate",
        description="Verify damage claims from CSV + images via Google Gemini cascade (OpenRouter fallback).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""quick start:
  python main.py check              # validate setup (no API calls)
  python main.py smoke              # 1 labeled sample claim + metrics
  python main.py evaluate --limit 5 # dev loop on sample set
  python main.py run --limit 2      # test claims preview
  python main.py run                # full output.csv for submission
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run (default) ---
    run_p = sub.add_parser(
        "run",
        help="Process claims CSV and write output.csv (submission pipeline)",
    )
    _add_common_args(run_p)
    run_p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input claims CSV (default: dataset/claims.csv)",
    )
    run_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: output.csv at repo root)",
    )
    run_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N claims (saves API calls during dev)",
    )

    # --- evaluate ---
    eval_p = sub.add_parser(
        "evaluate",
        help="Score predictions against labeled dataset/sample_claims.csv",
    )
    _add_common_args(eval_p)
    eval_p.add_argument(
        "--strategy",
        type=str,
        default="primary",
        help="Label for this run (used in output filenames)",
    )
    eval_p.add_argument("--limit", type=int, default=None, help="First N sample rows only")
    eval_p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics and predictions (default: evaluation/runs/)",
    )
    eval_p.add_argument(
        "--record-iteration",
        action="store_true",
        help="Archive run under evaluation/iterations/ and update registry.json",
    )
    eval_p.add_argument("--notes", type=str, default="", help="Notes for iteration snapshot")

    # --- smoke ---
    smoke_p = sub.add_parser(
        "smoke",
        help="Quick check: 1 sample claim + metrics (minimal API usage)",
    )
    _add_common_args(smoke_p)
    smoke_p.add_argument(
        "--record-iteration",
        action="store_true",
        help="Archive this smoke run as an iteration",
    )

    # --- check ---
    check_p = sub.add_parser("check", help="Validate .env, dataset files, and image paths (no API)")
    check_p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing dataset/",
    )

    # --- iterations ---
    sub.add_parser("iterations", help="List recorded evaluation iterations")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    known_commands = {"run", "evaluate", "smoke", "check", "iterations"}
    if not argv:
        argv = ["run"]
    elif argv[0] not in known_commands:
        if argv[0] in ("-h", "--help") and len(argv) == 1:
            pass
        else:
            argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    repo_root = default_repo_root().resolve()
    if getattr(args, "repo_root", None) is not None:
        repo_root = args.repo_root.resolve()
    load_env(repo_root)

    if command == "check":
        return check_setup(repo_root)

    if command == "iterations":
        return list_iterations()

    if command == "run":
        input_path = (args.input or repo_root / "dataset" / "claims.csv").resolve()
        output_path = (args.output or repo_root / "output.csv").resolve()
        run_predictions(
            repo_root=repo_root,
            input_path=input_path,
            output_path=output_path,
            model=args.model,
            limit=args.limit,
        )
        return 0

    if command == "evaluate":
        evaluate_sample(
            repo_root=repo_root,
            strategy=args.strategy,
            model=args.model,
            limit=args.limit,
            output_dir=args.output_dir,
            record_iteration_snapshot=args.record_iteration,
            notes=args.notes,
        )
        return 0

    if command == "smoke":
        evaluate_sample(
            repo_root=repo_root,
            strategy="smoke",
            model=args.model,
            limit=1,
            record_iteration_snapshot=args.record_iteration,
            notes="CLI smoke test (1 sample claim)",
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
