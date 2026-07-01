# Agentic RAG Evaluation on ViDoRe V3 Benchmark

This script ([`agentic_rag_eval.py`](agentic_rag_eval.py)) evaluates a
summarization + reasoning-based chunkless RAG approach using the
[ViDoRe V3](https://huggingface.co/datasets/vidore) benchmark. Unlike traditional
RAG pipelines that rely on similarity-based chunk retrieval, this method leverages
the structured layout of `DoclingDocument` objects, letting an LLM dynamically
interpret document hierarchy and navigate to the most relevant sections or pages.

## Overview

The evaluation pipeline consists of four steps:

1. **PDF Conversion** — Convert PDF files to `DoclingDocument` objects (JSON format).
2. **Heading Level Fixing** — Use `DoclingEditingAgent` to establish proper document hierarchy.
3. **Document Enrichment** — Add AI-generated summaries with `DoclingEnrichingAgent`.
4. **RAG Evaluation** — Perform reasoning-driven page selection and evaluate with NDCG@10.

## Key Features

- **Structure-Aware Retrieval** — Leverages document hierarchy instead of flat chunks.
- **In-Context Reasoning** — An LLM interprets document structure to find relevant sections.
- **Two Selector Algorithms** — Flat batch (`ReasoningBasedPageSelector`) or tree-guided
  (`TreeGuidedPageSelector`) page selection strategies.
- **Benchmark Evaluation** — Uses ViDoRe V3 for standardised page-level retrieval metrics.
- **Resumable Pipeline** — Each step skips already-processed files; Step 4 appends
  per-query results to a JSONL file so interrupted runs can be continued.

## Dataset

The script uses ViDoRe V3 benchmark datasets, which contain:

- **`pdfs/`** — Original documents (e.g., financial annual reports).
- **`queries/`** — Questions answered from those documents.
- **`corpus/`** — Page-level corpus with `corpus_id` ↔ `(doc_id, page_number)` mapping.
- **`qrels/`** — Ground truth relevance judgements (query → relevant pages).

Default dataset used in development: `vidore_v3_finance_en` (6 financial documents).

## Configuration

Use [`agentic_rag_eval_config.yaml`](agentic_rag_eval_config.yaml) as a starting
point. Copy it, set `dataset` to your local ViDoRe V3 directory, and adjust the
backend section to match your LLM server.

```yaml
# Path to ViDoRe V3 dataset (must contain pdfs/, queries/, corpus/, qrels/)
dataset: /path/to/vidore_v3_finance_en

# Output directory for all pipeline steps
output: ./scratch/agentic_rag_eval

# Summarization granularity
page_level: false   # false = element-level, true = page-level

# Backend
backend:
  type: lmstudio    # mellea | ollama | lmstudio | litellm | llama-server
  base_url: http://localhost:1234/v1
  timeout: 300
  temperature: 0.1
  models:
    reasoning: qwen2.5-7b-instruct-gguf
    writing: granite-4.1-3b
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `dataset` | *(required)* | Path to ViDoRe V3 dataset directory |
| `output` | `./scratch/agentic_rag_eval` | Root output directory for all steps |
| `page_level` | `false` | `false` = element-level enrichment; `true` = page-level |
| `summarization_style` | `sentences` | `sentences` → `meta.summary`; `keyphrases` → `meta.keywords` |
| `selector_algorithm` | `batch` | `batch` = `ReasoningBasedPageSelector`; `tree` = `TreeGuidedPageSelector` |
| `evaluation.top_k` | `10` | Pages retrieved per query (for NDCG@K) |
| `evaluation.batch_size` | `30` | Pages per reasoning iteration (batch selector) |
| `evaluation.early_stopping_threshold` | `0.95` | Confidence threshold for early stopping |
| `evaluation.max_iterations` | `8` | Max drill-down iterations (tree selector) |
| `backend.type` | — | LLM backend: `mellea`, `ollama`, `lmstudio`, `litellm`, `llama-server` |
| `backend.models.reasoning` | — | Model for editing and page selection |
| `backend.models.writing` | — | Model for summarisation |

## Usage

### Basic Usage

Run each step sequentially from the repository root:

```bash
# Step 1: Convert PDFs to DoclingDocument
python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 1

# Step 2: Fix heading levels
python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 2

# Step 3: Enrich with summaries
python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 3

# Step 4: Evaluate RAG — computes NDCG@10
python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 4
```

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML configuration file *(required)* |
| `--step {1,2,3,4}` | Which pipeline step to run *(required)* |
| `--dataset PATH` | Override dataset path from config |
| `--output PATH` | Override output directory from config |
| `--page-level` | Force page-level summarisation in Step 3 |
| `--summarization-style` | `sentences` or `keyphrases` (overrides config) |
| `--selector-algorithm` | `batch` or `tree` (overrides config) |

### Pipeline Steps in Detail

#### Step 1 — Convert PDFs to DoclingDocument

Converts all PDFs in `dataset/pdfs/` to `DoclingDocument` JSON using Docling's
`DocumentConverter`. Already-converted files are skipped.

**Output**: `<output>/step1_converted/*.json`

#### Step 2 — Fix Heading Levels

`DoclingEditingAgent` detects flat heading structures (all headings at the same
level, as is common with PDF extraction) and reassigns levels based on document
content. Afterwards `_hierarchize()` rebuilds the element tree.

**Output**: `<output>/step2_hierarchical/*.json` and `*.md`

#### Step 3 — Enrich with Summaries

Adds AI-generated annotations to each document. Two granularity modes:

**Element-level** (default, `page_level: false`):
- Generates a summary or keyphrase list for every section header, paragraph,
  and table individually.
- More granular; better for complex, fine-grained queries.

**Page-level** (`page_level: true`):
- Generates a single annotation per page plus a document-level summary derived
  from the actual text of the first three pages.
- Faster; recommended when running Step 4 with the `batch` selector.

Both modes save after each page (fault-tolerant resume).

**Output**: `<output>/step3_enriched/*.json`

#### Step 4 — Evaluate RAG with NDCG@10

Implements the full retrieval evaluation:

1. Loads enriched documents and their document-level summaries.
2. For each query, calls `select_relevant_documents` to identify candidate
   documents without peeking at ground truth.
3. For each candidate document, calls `select_pages` to rank the most relevant
   pages using the chosen selector algorithm.
4. If multiple documents are selected, `rerank_across_documents` merges results.
5. Converts Docling 1-indexed page numbers to ViDoRe 0-indexed corpus IDs.
6. Computes NDCG@10 with `ranx`, matching the official ViDoRe evaluation protocol.

Results are appended to a JSONL file per query, enabling seamless resume after
an interruption.

**Output**:
- `<output>/step4_evaluation/query_results.jsonl` — Per-query predictions and ground truth.
- `<output>/step4_evaluation/evaluation_results.json` — Aggregate NDCG@10 and timing.
- `<output>/step4_evaluation/retrieval_results.json` — Raw score map for external analysis.

## Architecture

### Traditional RAG vs. Agentic RAG

**Traditional RAG**:
1. Split document into fixed-size chunks.
2. Embed all chunks.
3. Find top-k chunks via vector similarity.
4. Generate answer from retrieved chunks.

**Agentic RAG (this approach)**:
1. Preserve document structure (sections, subsections, pages).
2. Generate per-section or per-page AI summaries.
3. LLM navigates document outline based on the query.
4. Retrieve only the relevant pages dynamically.
5. Optionally re-rank across multiple documents.

### Selector Algorithms

| Algorithm | Class | Description |
|-----------|-------|-------------|
| `batch` | `ReasoningBasedPageSelector` | Evaluates pages in flat batches; works with both element-level and page-level enrichment |
| `tree` | `TreeGuidedPageSelector` | Traverses the heading hierarchy top-down; requires element-level Step 3 enrichment |

## Evaluation Metric

**NDCG@10** (Normalised Discounted Cumulative Gain at 10) is the standard metric
for the ViDoRe benchmark. It measures both the relevance and the ranking position
of retrieved pages, enabling direct comparison with other retrieval methods.

## Output Structure

```
scratch/agentic_rag_eval/
├── step1_converted/
│   ├── bank_of_america_2024.json
│   └── ...
├── step2_hierarchical/
│   ├── bank_of_america_2024.json
│   ├── bank_of_america_2024.md
│   └── ...
├── step3_enriched/
│   ├── bank_of_america_2024.json
│   └── ...
└── step4_evaluation/
    ├── query_results.jsonl
    ├── evaluation_results.json
    └── retrieval_results.json
```

## Requirements

In addition to the standard `docling-agent` dependencies:

```bash
pip install ranx pandas pyarrow tqdm
```

- `docling` — PDF conversion.
- `docling-core` — `DoclingDocument` manipulation.
- `docling-agent` — Editing, enriching, and RAG agents.
- `ranx` — NDCG computation.
- `pandas` / `pyarrow` — Reading ViDoRe Parquet files.

## Performance Notes

- **Batch size** — Larger batches (40–50 pages) are faster but may exceed context
  limits or dilute the LLM's attention. Start with 30 and tune.
- **Model selection** — A capable reasoning model (Qwen 2.5 7B+, GPT-4-class) is
  important for quality retrieval decisions.
- **Context window** — Ensure your reasoning model has at least 8K tokens of context.
- **Fault tolerance** — Step 3 saves after every page; Step 4 flushes after every
  query. Both can be safely interrupted and resumed.

## References

- [ViDoRe Benchmark](https://huggingface.co/datasets/vidore)
- [Docling Documentation](https://www.docling.ai/)
- [DoclingAgent Repository](https://github.com/docling-project/docling-agent)
