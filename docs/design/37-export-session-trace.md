# Design — Export an agent trace tree to a file (generalize the RAG trace to all agents)

| | |
|---|---|
| **Issue** | [docling-agent#37](https://github.com/docling-project/docling-agent/issues/37) |
| **Author** | Pier-Jean Malandrino |
| **Date** | 2026-06-26 |
| **Status** | Proposed |
| **Scope** | `docling_agent/agent/agent_trace.py` (new), `docling_agent/agent/base.py`, `docling_agent/agent/rag_models.py`, `docling_agent/agent/rag.py`, `docling_agent/agent/orchestrator.py`, `docling_agent/task_model.py`, `docling_agent/cli/__init__.py`, public re-exports, tests |

## 1. Problem

`#39` gave `DoclingRAGAgent` a public `run_with_trace()` returning a typed `RAGTrace`
(per-document `RAGResult` → `RAGIteration`), so a consumer can inspect *what the RAG
agent did*. A maintainer's follow-up on that PR asked for the natural generalization: a
**similar tracing capability across all agents**, **exportable to a file** for debugging.
This is also issue #37 ("Logs: export session logs").

Today only RAG has a trace. The other agents (writer, editor, extractor, enricher,
orchestrator) expose nothing structured, and there is no way to persist a whole run —
spanning the orchestrator and the sub-agents it dispatches to — to a single file.

## 2. Goals

- [ ] Generalize the `#39` pattern into a **generic, typed trace** every agent produces.
- [ ] Make the orchestrator compose the traces of its sub-agents into a **tree** (one run = one tree).
- [ ] **Export the tree to a file** (JSON) for debugging.
- [ ] Keep `#39` intact: `RAGTrace` becomes a *specialization* of the generic trace; `DoclingRAGAgent.run_with_trace()` keeps its return type and behaviour.
- [ ] Be a **value-object** design — no global state, no concurrency hazard, no coupling to the logging subsystem.
- [ ] Zero behaviour change for existing callers of `run()` / `run_task()`; tracing is additive and opt-in.

## 3. Non-goals

- Capturing the *raw LLM prompts/responses* (the v1 idea). That is additive later: it is just two optional fields (`prompt`/`response`) on `AgentStep`. Keeping it out now avoids hooking the logging layer and keeps the design a clean value object. (See §10.)
- Rewriting each agent's internals to emit fine-grained steps in this PR. The base provides a useful default trace for every agent immediately; richer per-step detail is added agent-by-agent as a follow-up (RAG already has the richest form).
- Token/cost accounting (backends don't expose counts uniformly); `metadata` leaves room.
- A versioned on-disk schema. The file is the Pydantic JSON of the models; consumers pin the package version. Schema evolution is additive.

## 4. Context & constraints

- **The pattern already exists** — `#39`'s three layers (`RAGIteration` → `RAGResult` → `RAGTrace`) plus `run_with_trace()`. Generalizing means lifting that shape to the base class, not inventing a new mechanism.
- All agents derive from `BaseDoclingAgent` and share `run(task, document=None, sources=[], **kwargs) -> DoclingDocument` ([base.py](../../docling_agent/agent/base.py)).
- The orchestrator already dispatches to each sub-agent via its `run()` ([orchestrator.py](../../docling_agent/agent/orchestrator.py)); composing a tree means collecting each sub-agent's trace at the dispatch sites.
- `RAGTrace`/`RAGResult`/`RAGIteration` are Pydantic models ([rag_models.py](../../docling_agent/agent/rag_models.py)); making `RAGTrace` subclass the generic trace is backward-compatible as long as the generic fields have defaults.
- The CLI already wires a `LoggingConfig` from the task file ([cli/__init__.py](../../docling_agent/cli/__init__.py)); the file export hooks in there.

## 5. Proposed design

### 5.1 New models — `docling_agent/agent/agent_trace.py`

```python
class AgentStep(BaseModel):
    """One unit of work inside an agent run (a stage, operation, or iteration)."""
    name: str
    detail: str = ""
    output: str = ""
    duration_ms: int = 0
    model_id: str = ""
    metadata: dict[str, Any] = {}


class AgentTrace(BaseModel):
    """Trace of one agent run, with optional nested sub-agent traces (a tree)."""
    agent_type: str = ""
    task: str = ""
    steps: list[AgentStep] = []
    children: list[AgentTrace] = []        # sub-agent traces (orchestrator composition)
    duration_ms: int = 0
    model_id: str = ""
    result_name: str | None = None         # name of the produced DoclingDocument
    attachments: dict[str, Any] = {}       # domain-specific payloads
    output: DoclingDocument | None = Field(default=None, exclude=True)  # in-memory only

    def to_json(self, *, indent: int = 2) -> str: ...
    def save(self, path: str | Path) -> Path: ...
```

Design notes:
- **`children`** is what makes it "across all agents": the orchestrator nests the trace of each sub-agent it ran. One run → one tree → one file.
- **`output`** carries the produced `DoclingDocument` so `run_with_trace()` is the single source of truth (mirrors how `RAGTrace.final_answer` carries the RAG payload). It is `exclude=True`, so it is **never serialized** — at any nesting level — keeping the exported file small; `result_name` is the lightweight pointer that *is* persisted.
- All fields have defaults so subclassing is non-breaking (§5.3).

### 5.2 Base contract — every agent gets a trace for free

Add a concrete `run_with_trace()` to `BaseDoclingAgent`:

```python
def run_with_trace(self, task, document=None, sources=[], **kwargs) -> AgentTrace:
    start = time.perf_counter()
    result = self.run(task, document=document, sources=sources, **kwargs)
    return AgentTrace(
        agent_type=str(self.agent_type),
        task=task,
        duration_ms=int((time.perf_counter() - start) * 1000),
        model_id=self.get_reasoning_model_id(),
        result_name=getattr(result, "name", None),
        output=result,
    )
```

So **all five non-RAG agents expose a trace immediately**, with timing + model id + result, without bespoke code. `run()` stays the source of truth for the document. Agents with internal structure override `run_with_trace()` to add `steps` (RAG already does — §5.4).

### 5.3 `RAGTrace` becomes a specialization

```python
class RAGTrace(AgentTrace):
    query: str
    per_document: list[RAGResult]
    final_answer: str
```

`RAGTrace` *is-a* `AgentTrace`, so a RAG run drops straight into the tree under
`children`, and `DoclingRAGAgent.run_with_trace() -> RAGTrace` still satisfies the base
contract (covariant return). Because the generic fields default, existing construction
(`RAGTrace(query=..., per_document=..., final_answer=...)`) is unchanged — `#39` and its
tests keep passing.

`DoclingRAGAgent.run_with_trace()` is updated to also fill the generic fields
(`agent_type`, `task`, `model_id`, `result_name`) and to build the answer document onto
`output`; `run()` becomes `return run_with_trace(...).output` — tightening the single
source of truth `#39` introduced.

### 5.4 Orchestrator composes the tree

The orchestrator collects sub-agent traces during dispatch via a small instance hook
that is **inert unless tracing is active** (so `run_task()` is untouched and free):

```python
def _record_child(self, trace: AgentTrace) -> None:
    if self._child_traces is not None:
        self._child_traces.append(trace)

def run_task_with_trace(self, task: AgentTask) -> AgentTrace:
    self._child_traces = []
    start = time.perf_counter()
    try:
        doc = self.run_task(task)
        children = self._child_traces
    finally:
        self._child_traces = None
    return AgentTrace(
        agent_type="orchestrator", task=task.query, children=children,
        duration_ms=..., model_id=..., result_name=doc.name, output=doc,
    )
```

Each `_run_*` dispatch site changes from `return agent.run(...)` to:

```python
trace = agent.run_with_trace(...)
self._record_child(trace)
return cast(DoclingDocument, trace.output)
```

This is a small, mechanical change with no new control flow. When `run_task()` is called
directly (no trace), `_child_traces is None` and `_record_child` is a no-op.

### 5.5 Config & CLI

Add `trace_path: Path | None = None` to `LoggingConfig`. In the CLI run path, when set:

```python
trace = orchestrator.run_task_with_trace(agent_task)
trace.save(trace_path)
result = trace.output
```

### 5.6 Public exports

Re-export `AgentTrace` and `AgentStep` from `docling_agent.agents` and
`docling_agent.__init__` (alongside the existing `RAGTrace`/`RAGResult`/`RAGIteration`).

## 6. Alternatives considered

| Alternative | Why rejected |
|---|---|
| Ambient global/`ContextVar` recorder hooked into `log_llm_request/response` (the v1 of this PR) | Introduces global mutable state and couples tracing to the logging subsystem; concurrency-hazardous for a server consumer (Docling Studio). The value-object tree has none of these problems and mirrors `#39`. |
| New parallel trace type unrelated to `RAGTrace` | Duplicates the `#39` model and forces consumers to handle two shapes. Subclassing reuses it. |
| Make `run_with_trace()` abstract and rewrite every agent now | Large, risky, and unnecessary: the base default already gives every agent a useful trace; richer steps land incrementally. |
| Put the produced `DoclingDocument` in the serialized trace | Bloats the file and duplicates the saved output. `output` is in-memory only (`exclude=True`); `result_name` is the persisted pointer. |
| Return `(document, trace)` tuples | Breaks the `run()` contract and is clumsy to thread through the orchestrator. `AgentTrace.output` keeps a single return value. |

## 7. API contract

### Public surface added

```python
class AgentStep(BaseModel): ...
class AgentTrace(BaseModel):
    def to_json(self, *, indent: int = 2) -> str: ...
    def save(self, path: str | Path) -> Path: ...

BaseDoclingAgent.run_with_trace(task, document=None, sources=[], **kwargs) -> AgentTrace
DoclingOrchestratorAgent.run_task_with_trace(task: AgentTask) -> AgentTrace
# RAGTrace is now a subclass of AgentTrace.
```

### Invariants

- `agent.run(task, ...) ` and `agent.run_with_trace(task, ...).output` produce the same document.
- `run_with_trace`/`run_task_with_trace` add no behaviour change to `run`/`run_task`; called directly, `run_task()` records nothing.
- `AgentTrace.output` is never present in `to_json()` / `save()` output, at any depth; `result_name` is.
- A reloaded trace equals the dumped form (output excluded both ways).
- `len(tree.children)` equals the number of sub-agents the orchestrator dispatched to during the run, in dispatch order.
- `isinstance(rag_agent.run_with_trace(...), AgentTrace)` is `True`; `RAGTrace`'s `#39` fields and construction are unchanged.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Subclassing `RAGTrace` breaks `#39` construction/tests | All generic fields default; verified by keeping `#39`'s tests untouched and green. |
| Serializing a `DoclingDocument` accidentally | `output` is `exclude=True`; covered by a test asserting `output` absent and `result_name` present. |
| Orchestrator refactor changes `run_task()` behaviour | `_record_child` is a no-op when not tracing; the dispatch change is `run()` → `run_with_trace().output`, same document. Existing orchestrator tests are the regression net. |
| Default trace is "shallow" for non-RAG agents | Acceptable and documented: timing/model/result now exist for all; per-step detail is an additive follow-up per agent. |
| Memory: `output` holds full docs in the tree | In-memory only and short-lived; never serialized. |

## 9. Testing strategy

`tests/test_agent_trace.py`:
1. **Base default** — a minimal concrete agent: `run_with_trace()` returns an `AgentTrace` with `agent_type`, `task`, `model_id`, `result_name`, `output`, `duration_ms >= 0`.
2. **Equivalence** — `run()` and `run_with_trace().output` return the same document.
3. **RAG specialization** — `run_with_trace()` returns a `RAGTrace` that is an `AgentTrace`, with generic + RAG fields filled and the answer doc on `output`; `run()` still returns it.
4. **Serialization** — `output` excluded from `to_json()`, `result_name` present.
5. **Save round-trip** — nested tree (`children` + `steps`) saves and re-parses equal.
6. **Orchestrator tree** — `run_task_with_trace()` nests recorded children in dispatch order, with `output`/`result_name` set.
7. **No-op off** — `_record_child` outside a traced run does nothing.
8. **Regression** — `#39`'s `tests/test_rag_trace.py` untouched and green; orchestrator tests updated only where the dispatch now routes through `run_with_trace` (fake signatures widened to the real `run` signature).

## 10. Rollout

- Single PR linked to issue #37 (`Closes #37`); follows up `#39`.
- Pure addition + an internal refactor (RAG `run`/`run_with_trace`, orchestrator dispatch); no migration, no breaking change.
- Minor version bump on next release (new public API).
- Follow-ups (separate PRs): richer `steps` per agent (enricher/writer/editor/extractor); optional raw LLM `prompt`/`response` on `AgentStep`; Docling Studio consuming the exported tree.

## 11. References

- Issue: <https://github.com/docling-project/docling-agent/issues/37>
- Builds on: `#39` (`run_with_trace`), issue [#26](https://github.com/docling-project/docling-agent/issues/26)
- Base + agents: [base.py](../../docling_agent/agent/base.py), [orchestrator.py](../../docling_agent/agent/orchestrator.py)
- RAG models: [rag_models.py](../../docling_agent/agent/rag_models.py)
- Public exports: [docling_agent/__init__.py](../../docling_agent/__init__.py), [docling_agent/agents.py](../../docling_agent/agents.py)
