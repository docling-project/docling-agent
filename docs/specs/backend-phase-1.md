# Backend Phase 1 Specification

## Scope

Phase 1 establishes the backend abstraction layer and the new task configuration shape without migrating the agent runtime away from its current Mellea-oriented execution flow.

This phase introduces:

- `docling_agent/backends/`
- `BaseBackend`
- backend factory and registry
- `MelleaBackend`
- placeholder direct backends for `ollama`, `lmstudio`, and `litellm`
- top-level task `backend` configuration with nested model roles

This phase does not yet introduce:

- agent migration to `backend.instruct(...)`
- direct execution logic for `ollama`, `lmstudio`, or `litellm`
- removal of existing `agent_models.py` session helpers

## Runtime model

### Base contract

All backends implement:

```python
def instruct(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    requirements: list[Requirement] | None = None,
    retry_budget: int = 1,
) -> str
```

Shared semantics:

- `prompt` is the main instruction payload
- `model` is backend-specific model selection
- `system_prompt` is optional setup context
- `requirements` are accepted uniformly across all backends
- `retry_budget` belongs to the backend implementation

### Backend-specific expectations

#### `MelleaBackend`

- uses `mellea` internally
- preserves current requirement handling semantics
- preserves current retry delegation semantics

#### `OllamaBackend`

- no internal `mellea` usage
- currently a placeholder
- accepts `requirements` but ignores them for now

#### `LMStudioBackend`

- no internal `mellea` usage
- currently a placeholder
- accepts `requirements` but ignores them for now

#### `LiteLLMBackend`

- no internal `mellea` usage
- currently a placeholder
- accepts `requirements` but ignores them for now

## Configuration model

Task YAML now supports a top-level backend block:

```yaml
backend:
  type: mellea
  base_url:
  timeout:
  api_key_env:
  options: {}
```

Fields:

- `type`
  - one of `mellea`, `ollama`, `lmstudio`, `litellm`
- `base_url`
  - optional endpoint override
- `timeout`
  - optional request timeout in seconds
- `api_key_env`
  - optional environment variable name for credential lookup
- `options`
  - backend-specific free-form settings

Backend-scoped model role configuration lives inside the backend block:

```yaml
backend:
  type: ...
  models:
    reasoning: ...
    writing: ...
```

## Factory behavior

`create_backend(config)` resolves `config.type` through the registry and instantiates the matching backend.

Required properties:

- backend selection is centralized
- backend instantiation is deterministic
- unsupported backend names fail fast with a clear error

## Phase boundary

Phase 1 is complete when:

- the backend package exists
- task loading supports the new `backend` block
- the factory can construct all four backend classes
- `MelleaBackend` can execute an instruction call
- the direct backends exist as explicit placeholders
- docs and tests cover the new abstraction surface

Phase 1 is not complete only when direct non-Mellea execution exists; that belongs to a later phase.
