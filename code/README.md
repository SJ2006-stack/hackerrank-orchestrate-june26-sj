# Damage Claim Evidence Verifier

Python CLI for the HackerRank Orchestrate multi-modal evidence review challenge.

## Architecture

> **For judges:** One VLM call per claim, deterministic postprocess rules, Google-first model cascade with OpenRouter fallback, disk response cache, and image resize before upload. Tradeoffs below explain cost/latency vs accuracy choices.

Single vision-language model call per claim — no multi-agent or critic pipeline:

```text
CSV row + images → relevant_evidence_requirements (filtered)
                 → build_verification_prompt (1 user message + N images)
                 → vision_json_completion_routed (Google cascade → OpenRouter)
                 → apply_postprocess (deterministic policy layer)
                 → output.csv row
```

**Design tradeoffs:**

| Choice | Why |
|--------|-----|
| One VLM call per claim | Minimizes API cost and latency vs per-image agents |
| Google-first cascade | Gemma 4 + Gemini Flash on AI Studio quota; OpenRouter last |
| `apply_postprocess` rules | Fixes enum drift, evidence caps, injection flags — zero API cost |
| Disk response cache | Repeat smoke/eval on same data ≈ 0 new tokens |
| `RESIZE_MAX_PX=1024` | Cuts image token volume ~30–50% with minor quality loss |
| Smart evidence filtering | Sends only claim-relevant requirements to shrink prompt text |
| `temperature=0` | Reproducible outputs where the model supports it |

## Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env    # add GEMINI_API_KEY (+ OPENROUTER_API_KEY for fallback)
```

## Model routing

Per claim, the verifier tries models **in order** until one succeeds:

| Tier | Provider | Models |
|------|----------|--------|
| 1 | Google Gemini API | `gemma-4-26b-a4b-it` → `gemma-4-31b-it` |
| 2 | Google Gemini API | `gemini-2.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.5-flash` |
| 3 | OpenRouter (fallback) | `openrouter/free` |

Pin a single model with `--model <id>` (Google model ID or OpenRouter slug). Override the full cascade with `MODEL_ROUTING_ORDER` in `.env` (format: `google:model_id,openrouter:slug`).

Full command reference: **[CLI.md](CLI.md)**

## Response cache

Vision responses are cached on disk under `code/.cache/` (configurable via `CACHE_DIR`). Cache key = SHA256 of `(provider, model_id, system prompt hash, user text hash, per-image content hashes)`.

- Enabled by default (`CACHE_ENABLED=1`)
- Set `CACHE_ENABLED=0` for a fresh `run` submission if needed
- Second `smoke` or `evaluate` on unchanged data should show `cache_hits` ≈ row count in metrics JSON

## CLI commands

All commands run from the `code/` directory:

| Command | API calls | Purpose |
|---------|-----------|---------|
| `python main.py check` | **None** | Validate `.env`, dataset files, image paths |
| `python main.py smoke` | **1** sample claim | Fastest end-to-end sanity check |
| `python main.py evaluate --limit 5` | 5 sample claims | Dev loop with accuracy metrics |
| `python main.py run --limit 2` | 2 test claims | Preview test-set predictions |
| `python main.py run` | All test claims | Write submission `output.csv` |
| `python main.py iterations` | None | List archived iteration runs |

```bash
python main.py --help
python main.py run --help
python main.py evaluate --help
```

### Examples

```bash
# 1) No API — confirm setup
python main.py check

# 2) Minimal API — one labeled claim + latency + accuracy
python main.py smoke

# 3) Dev iteration (save snapshot for comparison)
python main.py evaluate --strategy google_cascade --limit 5 --record-iteration --notes "routing v1"

# 4) Full labeled evaluation
python main.py evaluate --strategy baseline_v1 --record-iteration

# 5) Submission output
python main.py run

# 6) Pin a single Google model
python main.py evaluate --model gemini-3.1-flash-lite --limit 3
```

Backward compatible: `python main.py --limit 1` is the same as `python main.py run --limit 1`.

Evaluation entry point still works: `python evaluation/main.py --strategy baseline_v1`.

## Iteration tracking

Use `--record-iteration` on `evaluate` or `smoke` to archive each run:

```text
evaluation/iterations/
├── registry.json
└── iter_001_baseline_v1/
    ├── summary.json
    ├── per_claim_timings.json
    ├── per_request_usage.json   # tokens, provider, model per claim
    ├── metrics_*.json
    └── mismatches_*.csv
```

`per_request_usage.json` records per claim: `provider`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_s`, `fallback_attempts`, and `models_tried`.

List runs: `python main.py iterations`

## Environment

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | **Required.** Google AI Studio key for Gemma 4 + Gemini Flash |
| `OPENROUTER_API_KEY` | Optional fallback when all Google models fail |
| `OPENROUTER_BASE_URL` | Default `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Default OpenRouter slug when pinning via `--model` |
| `MODEL_ROUTING_ORDER` | Optional comma-separated `provider:model_id` cascade override |
| `REQUEST_TIMEOUT_S` | Per-request API timeout (default `90`) |
| `CACHE_ENABLED` | `1` (default) enables disk cache; `0` disables |
| `CACHE_DIR` | Cache directory (default `code/.cache`) |
| `RESIZE_MAX_PX` | Max image edge in pixels before upload (default `1024`; empty = no resize) |

## Project layout

```text
code/
├── main.py           # Unified CLI (run | evaluate | smoke | check | iterations)
├── model_router.py   # Google-first cascade definitions
├── google_client.py  # Google Gemini API vision + JSON
├── llm_client.py     # Routed completion (Google + OpenRouter)
├── response_cache.py # Disk cache for vision responses
├── runner.py         # Shared pipeline logic
├── verifier.py       # Vision verification + cache stats
├── postprocess.py    # Deterministic policy layer after model output
├── data_loader.py    # CSV loading + smart evidence filtering
├── evaluation/
│   ├── main.py       # Alternate evaluation entry point
│   ├── metrics.py    # Strict + primary field accuracy (set-based flags)
│   ├── iterations/   # Archived iteration snapshots
│   └── runs/         # Latest per-strategy artifacts
└── ...
```

## Notes

- Missing images or API failures return conservative `not_enough_information` rows.
- Token usage and `cache_hits` are aggregated in CLI stats and saved in iteration artifacts.
- Use `--limit` while iterating to conserve Google AI Studio quotas.
- Primary evaluation metrics (`primary_field_accuracy`, `primary_exact_row_match_rate`) exclude justification/reason fields and compare `risk_flags` / `supporting_image_ids` as unordered sets.
