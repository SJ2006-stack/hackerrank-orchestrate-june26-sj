# CLI Reference

All commands run from the `code/` directory with the virtualenv active:

```bash
cd code
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Entry point: [`main.py`](main.py)  
Alternate evaluation entry: [`evaluation/main.py`](evaluation/main.py)

---

## Quick reference

| Command | API calls | What it does |
|---------|-----------|--------------|
| `python main.py check` | **0** | Validate `.env`, datasets, and that a sample image resolves |
| `python main.py smoke` | **1** | Run 1 labeled sample claim + accuracy metrics |
| `python main.py evaluate` | **All sample rows** (20) | Score against full `dataset/sample_claims.csv` |
| `python main.py evaluate --limit N` | **N** | Score first N sample rows only (dev loop) |
| `python main.py run` | **All test rows** (44) | Produce submission `output.csv` from `dataset/claims.csv` |
| `python main.py run --limit N` | **N** | Process first N test rows only |
| `python main.py iterations` | **0** | List archived evaluation runs |

---

## Global behavior

### Default command

- `python main.py` → same as `python main.py run`
- `python main.py --limit 2` → same as `python main.py run --limit 2`

### Help

```bash
python main.py --help
python main.py run --help
python main.py evaluate --help
python main.py smoke --help
python main.py check --help
```

### Model routing (default)

Unless `--model` is set, each claim tries models in order until one succeeds:

| Step | Provider | Model |
|------|----------|-------|
| 1 | Google | `gemma-4-26b-a4b-it` |
| 2 | Google | `gemma-4-31b-it` |
| 3 | Google | `gemini-2.5-flash-lite` |
| 4 | Google | `gemini-3.1-flash-lite` |
| 5 | Google | `gemini-3.5-flash` |
| 6 | OpenRouter | `openrouter/free` (fallback only) |

Pin one model:

```bash
python main.py evaluate --model gemini-3.1-flash-lite --limit 3
python main.py run --model openrouter/free --limit 1
```

Override cascade in `.env`:

```bash
MODEL_ROUTING_ORDER=google:gemma-4-26b-a4b-it,google:gemini-3.1-flash-lite,openrouter:openrouter/free
```

---

## `check` — setup validation (no API)

```bash
python main.py check
python main.py check --repo-root /path/to/repo
```

**Does:**

- Confirms `.env` exists
- Checks `GEMINI_API_KEY` is set (required)
- Warns if `OPENROUTER_API_KEY` is missing (fallback disabled)
- Verifies dataset CSVs exist
- Resolves the first sample image path on disk

**Use when:** First setup, after editing `.env`, before spending API quota.

---

## `smoke` — minimal end-to-end test (1 API call)

```bash
python main.py smoke
python main.py smoke --record-iteration
python main.py smoke --model gemma-4-26b-a4b-it
```

| Flag | Description |
|------|-------------|
| `--repo-root` | Repo root (default: parent of `code/`) |
| `--model` | Pin single model instead of full cascade |
| `--record-iteration` | Save snapshot under `evaluation/iterations/` |

**Does:**

- Runs **1 row** from `dataset/sample_claims.csv` (first row, `user_001`)
- Prints per-claim latency, tokens, model used, and `claim_status`
- Writes metrics to `evaluation/runs/metrics_smoke.json`
- Writes mismatches to `evaluation/runs/mismatches_smoke.csv`

**Use when:** Confirm API keys, routing, and vision pipeline work (~40–60s per claim).

---

## `evaluate` — labeled sample set (dev + full sample)

### Full sample (all 20 rows)

Omit `--limit` to evaluate the entire `dataset/sample_claims.csv`:

```bash
python main.py evaluate --strategy google_cascade --record-iteration
python main.py evaluate --strategy baseline_v1 --record-iteration --notes "full sample run"
```

### Dev loop (first N rows)

```bash
python main.py evaluate --strategy dev_v1 --limit 5
python main.py evaluate --strategy dev_v1 --limit 5 --record-iteration --notes "prompt tweak"
```

### Pin a single model

```bash
python main.py evaluate --model gemini-3.1-flash-lite --limit 3 --strategy pinned_31lite
```

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `primary` | Label used in output filenames |
| `--limit` | *(none = all rows)* | First N sample rows only |
| `--repo-root` | auto | Repository root |
| `--model` | full cascade | Pin single Google model ID or OpenRouter slug |
| `--output-dir` | `evaluation/runs/` | Where artifacts are written |
| `--record-iteration` | off | Archive run to `evaluation/iterations/iter_NNN_<strategy>/` |
| `--notes` | `""` | Free-text note stored in iteration summary |

**Does:**

- Reads `dataset/sample_claims.csv`
- Sends **all images** per row (`;`-separated paths in `image_paths`)
- Compares predictions to expected label columns
- Prints `claim_status_accuracy`, per-field accuracy, token totals, latency
- Writes:
  - `evaluation/runs/sample_predictions_<strategy>.csv`
  - `evaluation/runs/metrics_<strategy>.json`
  - `evaluation/runs/mismatches_<strategy>.csv`
  - `evaluation/runs/claim_status_confusion_<strategy>.json`

**Use when:** Improving prompts/postprocess; compare strategies before full test run.

---

## `run` — submission pipeline (test set)

### Full test set (all 44 rows)

```bash
python main.py run
python main.py run --output ../output.csv
```

### Preview (first N rows)

```bash
python main.py run --limit 2
python main.py run --limit 1 --model gemma-4-26b-a4b-it
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `dataset/claims.csv` | Input claims CSV |
| `--output` | `output.csv` (repo root) | Predictions output path |
| `--limit` | *(none = all rows)* | First N test rows only |
| `--repo-root` | auto | Repository root |
| `--model` | full cascade | Pin single model |

**Does:**

- Processes each row in `dataset/claims.csv`
- Loads user history and evidence requirements per claim
- Calls vision model(s) with conversation + all case images
- Writes `output.csv` with the required 14-column schema

**Use when:** Generating final submission file (after sample eval looks good).

---

## `iterations` — compare past runs (no API)

```bash
python main.py iterations
```

**Does:**

- Reads `evaluation/iterations/registry.json`
- Prints table: iteration ID, strategy, `claim_status` accuracy, avg latency, timestamp

**Use when:** Tracking progress across dev runs.

---

## Alternate entry: `evaluation/main.py`

Same as `python main.py evaluate` (no `smoke`, `run`, `check`, or `iterations`):

```bash
python evaluation/main.py --strategy baseline_v1
python evaluation/main.py --strategy google_cascade --limit 5 --record-iteration
python -m evaluation.main --strategy baseline_v1 --record-iteration
```

---

## Recommended workflow

```bash
# 1. Free — validate setup
python main.py check

# 2. One API call — routing + tokens
python main.py smoke --record-iteration

# 3. Dev loop (5 claims)
python main.py evaluate --strategy dev_v1 --limit 5 --record-iteration

# 4. Compare runs
python main.py iterations

# 5. Full sample (20 claims) when dev looks good
python main.py evaluate --strategy google_cascade --record-iteration

# 6. Submission (44 test claims)
python main.py run
```

---

## Output artifacts

### Per run (`evaluation/runs/`)

| File | Contents |
|------|----------|
| `metrics_<strategy>.json` | Accuracy (strict + primary), latency, token totals, `cache_hits`, model breakdown |
| `mismatches_<strategy>.csv` | Expected vs predicted for wrong rows |
| `sample_predictions_<strategy>.csv` | Full predictions |
| `claim_status_confusion_<strategy>.json` | Confusion matrix |

### Per iteration (`evaluation/iterations/iter_NNN_<strategy>/`)

| File | Contents |
|------|----------|
| `summary.json` | Metrics rollup (`primary_field_accuracy`, `primary_exact_row_match_rate`, `cache_hits`) + artifact paths |
| `per_claim_timings.json` | Latency and `image_count` per row |
| `per_request_usage.json` | `prompt_tokens`, `completion_tokens`, `model`, `provider`, `cache_hit` per row |
| Copies of metrics, mismatches, and predictions |

Example `summary.json` metrics block (after a cached re-run):

```json
{
  "metrics": {
    "rows": 20,
    "claim_status_accuracy": 0.85,
    "primary_exact_row_match_rate": 0.40,
    "primary_field_accuracy": {
      "claim_status": 0.85,
      "issue_type": 0.80,
      "risk_flags": 0.75
    },
    "cache_hits": 20,
    "model_calls": 0,
    "total_tokens": 0
  }
}
```

`cache_hits` counts claims served from disk cache (`CACHE_DIR`, default `code/.cache/`). When `CACHE_ENABLED=0`, expect `cache_hits: 0`.

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes | Google AI Studio — Gemma 4 + Gemini Flash |
| `OPENROUTER_API_KEY` | No | Fallback tier only |
| `OPENROUTER_BASE_URL` | No | Default `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | No | Default when pinning OpenRouter via `--model` |
| `MODEL_ROUTING_ORDER` | No | Override cascade (`provider:model_id,...`) |
| `REQUEST_TIMEOUT_S` | No | API timeout per attempt (default `90`) |
| `CACHE_ENABLED` | No | `1` (default) enables disk cache; `0` disables for fresh runs |
| `CACHE_DIR` | No | Cache directory (default `code/.cache`) |
| `RESIZE_MAX_PX` | No | Max image edge before upload (default `1024`; empty = no resize) |

### Cache behavior

- Cache is **on by default** (`CACHE_ENABLED=1`). Responses stored under `code/.cache/` as JSON keyed by prompt + image hashes.
- On a cache hit, `per_request_usage.json` shows `cache_hit: true`, zero new tokens, and metrics JSON includes `cache_hits`.
- Run `smoke` twice with `--record-iteration` to confirm cache hits on the second run.
- Disable cache for submission runs if you need uncached model output: `CACHE_ENABLED=0 python main.py run`.

See [`.env.example`](../.env.example) at the repo root.
