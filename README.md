# Docling Agent

[![CI](https://github.com/docling-project/docling-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/docling-project/docling-agent/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/docling-agent)](https://pypi.org/project/docling-agent/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-agent)](https://pypi.org/project/docling-agent/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-agent)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling-agent/month)](https://pepy.tech/projects/docling-agent)
[![Chat with Dosu](https://dosu.dev/dosu-chat-badge.svg)](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling Agent is a Python library for AI-powered document workflows — writing, editing, extracting structured data, and enriching documents with metadata.

> [!NOTE]
> **This package is under active development. Feedback, suggestions, and contributions are very welcome.**

## Features

- [Document writing](examples/example_01_write_report.py): Generate well-structured reports from natural prompts and export to JSON/Markdown/HTML.
- [Targeted editing](examples/example_02_edit_report.py): Load an existing Docling JSON and apply focused edits with natural-language tasks.
- [Schema-guided extraction](examples/example_03_extract_schema.py): Extract typed fields from PDFs/images using a simple schema and produce HTML reports. See examples on curriculum_vitae, papers, invoices, etc.
- [Document enrichment](examples/example_04_enrich_document.py): Enrich existing documents with summaries, search keywords, key entities, and item classifications (language/function).
- Model-agnostic: Choose `mellea`, `ollama`, `lmstudio`, `litellm`, or `llama-server` through backend configuration.
- Simple API surface: Use `agent.run(...)` with `DoclingDocument` in/out; save via `save_as_*` helpers.
- [Run tracing](#trace-what-an-agent-did): Get timing, model and sub-agent traces for a run with `run_with_trace(...)`, and export a whole session to one JSON file.
- Optional tools: Integrate external tools (e.g., MCP) when available.

## Installation

```bash
pip install docling-agent
```

Requires Python 3.11 or higher.

## Getting Started

Each snippet shows how to initialise an agent, run a task, and save the result.

### Write a New Document

Generate well-structured reports from natural prompts and export to JSON, Markdown, or HTML ([example](examples/example_01_write_report.py)).

```python
from docling_agent.agents import BackendConfig, DoclingWritingAgent, ModelConfig, create_backend

backend = create_backend(
    BackendConfig(
        type="ollama",
        base_url="http://localhost:11434",
        models=ModelConfig(reasoning="qwen3:8b", writing="qwen3:8b"),
    )
)
agent = DoclingWritingAgent(backend=backend, tools=[])
doc = agent.run("Write a brief report on polymers in food packaging with a small comparison table.")
doc.save_as_html("./scratch/report.html")
```

### Edit an Existing Document

Use natural-language tasks to update a Docling Document ([example](examples/example_02_edit_report.py)). Run multiple tasks to iteratively refine content, structure, or formatting.

```python
from pathlib import Path
from docling_core.types.doc.document import DoclingDocument
from docling_agent.agents import BackendConfig, DoclingEditingAgent, ModelConfig, create_backend

ipath = Path("./examples/example_02_edit_resources/20250815_125216.json")
doc = DoclingDocument.load_from_json(ipath)

backend = create_backend(
    BackendConfig(
        type="mellea",
        models=ModelConfig(reasoning="OPENAI_GPT_OSS_20B", writing="OPENAI_GPT_OSS_20B"),
    )
)
agent = DoclingEditingAgent(backend=backend, tools=[])
updated = agent.run(task="Put polymer abbreviations in a separate column in the first table.", document=doc)
updated.save_as_html("./scratch/updated_table.html")
```

### Extract Structured Data with a Schema

Define a simple schema and provide a list of files (PDFs/images); the agent produces an HTML report with extracted fields ([example](examples/example_03_extract_schema.py)).

```python
from pathlib import Path
from docling_agent.agents import BackendConfig, DoclingExtractingAgent, ModelConfig, create_backend

schema = {"invoice-number": "string", "total": "float", "currency": "string"}
sources = sorted([p for p in Path("./examples/example_03_extract/invoices").rglob("*.*") if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}])

backend = create_backend(
    BackendConfig(
        type="mellea",
        models=ModelConfig(reasoning="OPENAI_GPT_OSS_20B", writing="OPENAI_GPT_OSS_20B"),
    )
)
agent = DoclingExtractingAgent(backend=backend, tools=[])
report = agent.run(task=str(schema), sources=sources)
report.save_as_html("./scratch/invoices_extraction_report.html")
```

### Enrich an Existing Document

Run enrichment passes — summaries, keywords, entities, and classifications — on a Docling Document ([example](examples/example_04_enrich_document.py)).

```python
from pathlib import Path
from docling_core.types.doc.document import DoclingDocument
from docling_agent.agents import BackendConfig, DoclingEnrichingAgent, ModelConfig, create_backend

ipath = Path("./examples/example_02_edit_resources/20250815_125216.json")
doc = DoclingDocument.load_from_json(ipath)

backend = create_backend(
    BackendConfig(
        type="mellea",
        models=ModelConfig(reasoning="OPENAI_GPT_OSS_20B", writing="OPENAI_GPT_OSS_20B"),
    )
)
agent = DoclingEnrichingAgent(backend=backend, tools=[])
enriched = agent.run(task="Summarize each paragraph, table, and section header.", document=doc)
enriched.save_as_html("./scratch/enriched_summaries.html")
```

### Trace What an Agent Did

Every agent has `run_with_trace()` next to `run()`. It returns an `AgentTrace` (timing, model and
the produced document) instead of just the document. The orchestrator uses `run_task_with_trace()`,
which nests the trace of each sub-agent it ran, so a whole session exports to one JSON file.

```python
from docling_agent.agents import BackendConfig, DoclingOrchestratorAgent, RAGTask, create_backend

orchestrator = DoclingOrchestratorAgent(backend=create_backend(BackendConfig(type="mellea")), tools=[])
trace = orchestrator.run_task_with_trace(RAGTask(query="What is the conclusion?", sources=["./report.pdf"]))

print(trace.duration_ms, [c.agent_type for c in trace.children])  # 8421 ['enricher', 'rag']
trace.save("./scratch/trace.json")
answer = trace.output
```

To export from a task file instead, set `logging.trace_path`:

```yaml
logging:
  trace_path: ./scratch/trace.json
```

## Backend Configuration

Task files select the backend via an explicit `backend` block:

```yaml
backend:
  type: ollama  # mellea | ollama | lmstudio | litellm | llama-server
  base_url: http://localhost:11434
  timeout: 120
  models:
    reasoning: qwen3:8b
    writing: qwen3:8b
```

Typical defaults:

- `mellea`: model names like `OPENAI_GPT_OSS_20B`
- `ollama`: model names like `qwen3:8b`
- `lmstudio`: model names like `granite-3.3-8b-instruct`
- `litellm`: routed model names like `openai/gpt-4.1-mini`
- `llama-server`: GGUF model names as loaded by llama.cpp's `llama-server` (default `http://localhost:8080/v1`)

## Examples

Explore the [`examples/`](examples/) folder for end-to-end scripts covering document writing, editing, extraction, enrichment, RAG querying, and more.

## Technical Report

For more details on Docling's inner workings, check out the [Docling Technical Report](https://arxiv.org/abs/2408.09869).

## Contributing

Please read [Contributing to Docling Agent](CONTRIBUTING.md) for details.

## References

If you use Docling or Docling Agent in your projects, please consider citing the following:

```bib
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}
```

## License

The Docling Agent codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.
