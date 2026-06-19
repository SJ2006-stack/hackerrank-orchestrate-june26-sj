# Evaluation iterations

Each run of `python -m evaluation.main` with `--record-iteration` saves a snapshot here.

## Layout

```text
iterations/
├── registry.json              # Index of all iterations (append-only)
├── iter_001_baseline_v1/
│   ├── summary.json           # Metrics + latency summary
│   ├── per_claim_timings.json # Per-row model latency
│   ├── metrics_*.json
│   ├── sample_predictions_*.csv
│   ├── mismatches_*.csv
│   └── claim_status_confusion_*.json
└── iter_002_...
```

## Compare runs

```bash
cat evaluation/iterations/registry.json
```

Open `mismatches_<strategy>.csv` inside each `iter_*` folder to see what changed between iterations.
