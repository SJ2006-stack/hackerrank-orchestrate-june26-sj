# Evaluation report

Fill this in after running at least two strategy comparisons on `dataset/sample_claims.csv`, then document operational metrics from a full `main.py run` on `dataset/claims.csv`.

## Strategy comparison

| Strategy | Model | claim_status accuracy | primary exact row match | exact row match (strict) | Notes |
|---|---|---:|---:|---:|---|
| `google_cascade` | Gemma/Gemini cascade → `openrouter/free` | TBD | TBD | TBD | Default routing (Google first) |
| `pinned_google` | TBD (e.g. `gemini-3.1-flash-lite` via `--model`) | TBD | TBD | TBD | Single Google model pin |
| `pinned_free` | `openrouter/free` via `--model` | TBD | TBD | TBD | OpenRouter-only fallback test |

### Key aggregate metrics

Fill from `evaluation/runs/metrics_<strategy>.json` or `evaluation/iterations/iter_NNN_<strategy>/summary.json` → `metrics`.

| Metric | `google_cascade` | `pinned_google` | `pinned_free` |
|---|---:|---:|---:|
| `primary_field_accuracy` (mean over PRIMARY_FIELDS) | TBD | TBD | TBD |
| `primary_exact_row_match_rate` | TBD | TBD | TBD |
| Cache hit rate (`cache_hits / rows`) | TBD | TBD | TBD |

### Primary field accuracy (relaxed — flags/IDs as sets)

| Field | `google_cascade` | `pinned_google` |
|---|---|---|
| `claim_status` | TBD | TBD |
| `issue_type` | TBD | TBD |
| `object_part` | TBD | TBD |
| `severity` | TBD | TBD |
| `evidence_standard_met` | TBD | TBD |
| `valid_image` | TBD | TBD |
| `risk_flags` (set) | TBD | TBD |
| `supporting_image_ids` (set) | TBD | TBD |

### Per-field accuracy — strict (all prediction columns)

| Field | `google_cascade` | `pinned_google` |
|---|---|---|
| `claim_status` | TBD | TBD |
| `issue_type` | TBD | TBD |
| `object_part` | TBD | TBD |
| `severity` | TBD | TBD |
| `risk_flags` | TBD | TBD |

## Final strategy for test set

- **Chosen strategy:** TBD
- **Why:** TBD

## Operational analysis

Document assumptions after running `evaluation/main.py` (both strategies) and full `main.py` on test claims. Google Gemini API quotas apply to the primary cascade; OpenRouter free tier is last-resort fallback.

### Usage summary

| Metric | Sample set (`sample_claims.csv`) | Full test set (`claims.csv`) |
|---|---:|---:|
| Model calls | TBD | TBD |
| Cache hits | TBD | TBD |
| Images processed | TBD | TBD |
| Input tokens (`prompt_tokens`) | TBD | TBD |
| Output tokens (`completion_tokens`) | TBD | TBD |
| Total tokens | TBD | TBD |
| Providers used | TBD | TBD |
| Models used | TBD | TBD |
| Fallback attempts | TBD | TBD |
| Approx. cost (USD) | TBD | TBD |
| Wall-clock runtime | TBD | TBD |
| Avg. latency per claim | TBD | TBD |

Per-request detail: `evaluation/iterations/<id>/per_request_usage.json`.  
Aggregate metrics: `evaluation/iterations/<id>/metrics_*.json` → `primary_field_accuracy`, `primary_exact_row_match_rate`, `cache_hits`.

### Rate limits, TPM, and reliability

| Topic | Google cascade | OpenRouter fallback |
|---|---|---|
| TPM / RPM limits observed | TBD | TBD |
| Throttling or queue delays | TBD | TBD |
| Retries / backoff | TBD | TBD |
| Cache hit rate (`code/.cache/`) | TBD | TBD |

### Pricing assumptions

| Item | Value |
|---|---|
| Primary provider | Google Gemini API (`GEMINI_API_KEY`) |
| Fallback provider | OpenRouter (`openrouter/free`) |
| Input price | TBD (from Google AI Studio / OpenRouter pricing) |
| Output price | TBD |
| Notes | Token counts recorded in `per_request_usage.json` for manual cost calc |

## How to reproduce

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure env at repo root
cp ../.env.example ../.env
# Edit ../.env with GEMINI_API_KEY, OPENROUTER_API_KEY (optional fallback)

# Smoke test (run twice to verify cache)
python main.py smoke --record-iteration
python main.py smoke --record-iteration

# Strategy comparison on sample set
python main.py evaluate --strategy google_cascade --record-iteration
python main.py evaluate --strategy pinned_google --model gemini-3.1-flash-lite --record-iteration

# Full test submission
python main.py run
```

Metrics JSON paths: `evaluation/runs/metrics_<strategy>.json`, `evaluation/iterations/iter_NNN_<strategy>/summary.json`.
