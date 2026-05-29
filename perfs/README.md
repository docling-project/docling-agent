# Performance Evaluation Utilities

This folder contains extraction evaluation scripts and tests that are useful for local benchmarking, but are intentionally kept outside the `docling_agent` package.

## Run Evaluation

From repository root:

```bash
python perfs/run_eval.py \
  --predictions /path/to/predictions \
  --ground-truth /path/to/ground_truth \
  --output perfs/result.json
```

Optional flags:

- `--fuzzy`: enable fuzzy string matching.
- `--fuzzy-threshold`: fuzzy match threshold in `[0.0, 1.0]` (default: `0.85`).

## Run Perf Tests

```bash
pytest perfs/test_eval.py
```
