# Evaluation

Delegates to the shared runner. Prefer the unified CLI:

```bash
cd code
python main.py evaluate --strategy baseline_v1 --limit 5 --record-iteration
python main.py smoke
python main.py iterations
```

Or use this entry point directly:

```bash
python -m evaluation.main --strategy baseline_v1 --limit 5 --record-iteration
```

## Outputs

Artifacts go to `evaluation/runs/` unless `--output-dir` is set.

| File | Description |
|---|---|
| `sample_predictions_<strategy>.csv` | Model predictions |
| `metrics_<strategy>.json` | Accuracy, latency, per-field breakdown |
| `mismatches_<strategy>.csv` | Expected vs predicted for wrong rows |
| `claim_status_confusion_<strategy>.json` | Confusion matrix |

With `--record-iteration`, a copy is archived under `evaluation/iterations/`.
