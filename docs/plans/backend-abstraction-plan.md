# Backend Abstraction Plan

## Goal

Refactor `docling-agent` so backend selection is explicit, centralized, and backend-agnostic at the agent layer, while still allowing `mellea` to remain one first-class backend option.

Supported backends:

1. `ollama`
2. `lmstudio`
3. `litellm`
4. `mellea`

## Core design decision

The abstraction should not hide that `mellea` is different.

- `MelleaBackend` is allowed to use `mellea` internally.
- `OllamaBackend`, `LMStudioBackend`, and `LiteLLMBackend` must not use any `mellea` code internally.
- all four backends share one `BaseBackend` interface
- `requirements` are accepted by the base interface for all backends
- only `MelleaBackend` uses those requirements initially
- the other three backends accept them but ignore them at first

This keeps the public contract uniform without forcing all backends to emulate Mellea behavior.

## Proposed module layout

```text
docling_agent/backends/
  __init__.py
  base.py
  factory.py
  registry.py
  mellea_backend.py
  ollama_backend.py
  openai_compatible.py
  lmstudio_backend.py
  litellm_backend.py
```

## Base interface

`BaseBackend` defines the shared runtime contract.

```python
from abc import ABC, abstractmethod
from mellea.stdlib.requirements import Requirement


class BaseBackend(ABC):
    backend_type: str

    @abstractmethod
    def instruct(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None = None,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        ...
```

Important behavior:

- the base interface accepts Mellea `Requirement` objects
- this is an intentional compatibility choice
- non-Mellea backends do not need to interpret them initially
- retry behavior belongs to each concrete backend, not to the agent layer

## Backend behavior

### `MelleaBackend`

Purpose:

- preserve current behavior with minimal regression risk

Behavior:

- uses `mellea` internally
- passes `requirements` through to Mellea
- delegates retry logic to Mellea rejection sampling

### `OllamaBackend`

Purpose:

- direct Ollama support without any Mellea dependency

Behavior:

- uses Ollama directly
- accepts `requirements` but ignores them initially
- owns its own retry logic
- first implementation may treat retry as a null operation

Future enhancement:

- requirements can later be rendered into prompt text

### `LMStudioBackend`

Purpose:

- direct LM Studio support without any Mellea dependency

Behavior:

- uses LM Studio's OpenAI-compatible API directly
- accepts `requirements` but ignores them initially
- owns its own retry logic
- first implementation may treat retry as a null operation

Future enhancement:

- requirements can later be rendered into prompt text

### `LiteLLMBackend`

Purpose:

- direct LiteLLM support without any Mellea dependency

Behavior:

- uses LiteLLM through its API directly
- accepts `requirements` but ignores them initially
- owns its own retry logic
- first implementation may treat retry as a null operation

Future enhancement:

- requirements can later be rendered into prompt text

## YAML config shape

Move backend selection into its own config block.

```yaml
query: "Summarize the document."
mode: enrich
sources:
  - ./paper.pdf

backend:
  type: ollama
  base_url: http://localhost:11434
  timeout: 120

models:
  reasoning: qwen3:8b
  writing: qwen3:8b
```

Examples:

```yaml
backend:
  type: lmstudio
  base_url: http://localhost:1234/v1
models:
  reasoning: granite-3.3-8b-instruct
  writing: granite-3.3-8b-instruct
```

```yaml
backend:
  type: litellm
  base_url: http://localhost:4000/v1
  api_key_env: LITELLM_API_KEY
models:
  reasoning: openai/gpt-4.1-mini
  writing: openai/gpt-4.1-mini
```

```yaml
backend:
  type: mellea
models:
  reasoning: OPENAI_GPT_OSS_20B
  writing: OPENAI_GPT_OSS_20B
```

## Refactor phases

### Phase 1: Introduce backend package

1. add `docling_agent/backends/`
2. define `BaseBackend`
3. add backend config models and factory
4. implement `MelleaBackend` first

### Phase 2: Move agent construction onto backend objects

1. update `BaseDoclingAgent` to hold a backend instance instead of Mellea model identifiers
2. remove backend creation from agent modules
3. centralize backend/model setup in one factory path

### Phase 3: Migrate the current agents

Update:

- `writer.py`
- `editor.py`
- `enricher.py`
- `extractor.py`
- `rag.py`
- `orchestrator.py`

Target state:

- agents call `backend.instruct(...)`
- agents pass through `requirements`
- agents do not create Mellea sessions directly

### Phase 4: Add direct backends

1. implement `OllamaBackend`
2. implement `LMStudioBackend`
3. implement `LiteLLMBackend`

Implementation note:

- `LMStudioBackend` and `LiteLLMBackend` should share an `OpenAICompatibleBackend` helper base

### Phase 5: Update docs and tests

1. update task YAML template
2. update README examples
3. add tests for backend factory
4. add backend smoke tests

## File impact

High-impact existing files:

- `docling_agent/task_model.py`
- `docling_agent/cli/__init__.py`
- `docling_agent/agent/base.py`
- `docling_agent/agent_models.py`
- `docling_agent/agent/orchestrator.py`
- `docling_agent/agent/writer.py`
- `docling_agent/agent/editor.py`
- `docling_agent/agent/enricher.py`
- `docling_agent/agent/extractor.py`
- `docling_agent/agent/rag.py`

New files:

- everything under `docling_agent/backends/`

## Acceptance criteria

- switching backend in YAML changes runtime behavior cleanly
- `MelleaBackend` preserves current Mellea-driven behavior
- `OllamaBackend`, `LMStudioBackend`, and `LiteLLMBackend` contain no internal `mellea` usage
- all backends accept `requirements`
- non-Mellea backends may ignore `requirements` initially
- retry logic lives inside each backend implementation
- agent modules no longer instantiate provider-specific sessions directly
