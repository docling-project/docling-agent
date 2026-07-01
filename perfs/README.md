# Performance Evaluation Utilities

This folder contains evaluation scripts and benchmarks that are useful for local
benchmarking, but are intentionally kept outside the `docling_agent` package.

---

## Extraction Quality Evaluation (`eval.py` / `run_eval.py`)

Measures field-level precision, recall and F1 between predicted and ground-truth
JSON extraction files.

### Run

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

### Tests

```bash
pytest perfs/test_eval.py
```

---

## Agentic RAG Evaluation (`agentic_rag_eval.py`)

Evaluates a summarization + reasoning-based chunkless RAG approach on the
[ViDoRe V3](https://huggingface.co/datasets/vidore) benchmark using **NDCG@10**.

Unlike traditional vector-similarity RAG this pipeline:

1. Preserves document structure via Docling conversion and heading-level normalisation.
2. Generates AI summaries at element or page level with `DoclingEnrichingAgent`.
3. Uses `ReasoningBasedPageSelector` or `TreeGuidedPageSelector` to navigate
   document structure and retrieve the most relevant pages per query.

See [`agentic_rag_eval_README.md`](agentic_rag_eval_README.md) for the full guide.

### Quick start

Copy and edit the config template, then run each pipeline step:

```bash
cp perfs/agentic_rag_eval_config.yaml perfs/my_config.yaml
# Edit my_config.yaml: set `dataset` to your ViDoRe dataset path

# Step 1: Convert PDFs → DoclingDocument JSON
python perfs/agentic_rag_eval.py --config perfs/my_config.yaml --step 1

# Step 2: Fix heading hierarchy
python perfs/agentic_rag_eval.py --config perfs/my_config.yaml --step 2

# Step 3: Enrich with AI summaries
python perfs/agentic_rag_eval.py --config perfs/my_config.yaml --step 3

# Step 4: Evaluate — computes NDCG@10
python perfs/agentic_rag_eval.py --config perfs/my_config.yaml --step 4
```

Additional dependencies required for Step 4:

```bash
pip install ranx pandas pyarrow
```
