[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-agent-project/docling-agent)](https://opensource.org/licenses/MIT)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

# Docling-Agent

Docling-agent simplifies agentic operation on documents, such as writing, editing, summarizing, etc.

> [!NOTE]
> **This package is still immature and work-in-progress. We are happy to get comments, suggestions, code contributions, etc!**

## Features

- [Document writing](examples/example_01_write_report.py): Generate well-structured reports from natural prompts and export to JSON/Markdown/HTML.
- [Targeted editing](examples/example_02_edit_report.py): Load an existing Docling JSON and apply focused edits with natural-language tasks.
- [Schema-guided extraction](examples/example_03_extract_schema.py): Extract typed fields from PDFs/images using a simple schema and produce HTML reports. See examples on curriculum_vitae, papers, invoices, etc.
- [Document enrichment](examples/example_04_enrich_document.py): Enrich existing documents with summaries, search keywords, key entities, and item classifications (language/function).
- Model-agnostic: Choose `mellea`, `ollama`, `lmstudio`, or `litellm` through backend configuration.
- Simple API surface: Use `agent.run(...)` with `DoclingDocument` in/out; save via `save_as_*` helpers.
- Optional tools: Integrate external tools (e.g., MCP) when available.

Quick start (writing):

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
doc = agent.run("Write a one-page summary about polymers in food packaging.")
doc.save_as_html("report.html")
```

## Installation

**Coming soon**

## Getting started

Below are three minimal, end-to-end examples mirroring the scripts in the examples folder. Each snippet shows how to initialize an agent, run a task, and save the result.

### Write a new document (see [example](examples/example_01_write_report.py)):

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

### Edit an existing document (see [example](examples/example_02_edit_report.py)):

Use natural-language tasks to update a Docling JSON. You can run multiple tasks to iteratively refine content, structure, or formatting.

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

### Extract structured data with a schema (see [example](examples/example_03_extract_schema.py)):

Define a simple schema and provide a list of files (PDFs/images). The agent produces an HTML report with extracted fields.

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

### Enrich an existing document (see [example](examples/example_04_enrich_document.py)):

Run enrichment passes like summaries, keywords, entities, and classifications on a Docling JSON.

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

## Backend Configuration

Task files now select the backend explicitly:

```yaml
backend:
  type: ollama  # mellea | ollama | lmstudio | litellm
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

## Documentation

**Coming soon**

## Examples

Go hands-on with our [examples](https://docling-project.github.io/docling/examples/),
demonstrating how to address different application use cases with Docling.

## Integrations

To further accelerate your AI application development, check out Docling's native
[integrations](https://docling-project.github.io/docling/integrations/) with popular frameworks
and tools.

## Get help and support

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling/discussions).

## Technical report

For more details on Docling's inner workings, check out the [Docling Technical Report](https://arxiv.org/abs/2408.09869).

## Contributing

Please read [Contributing to Docling](https://github.com/docling-project/docling/blob/main/CONTRIBUTING.md) for details.

## References

If you use Docling or Docling-agent in your projects, please consider citing the following:

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

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.
