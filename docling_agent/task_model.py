from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator


class OutputConfig(BaseModel):
    """Where and how to write the result."""

    path: Path | None = None
    format: Literal["markdown", "html", "json"] = "markdown"


class ModelConfig(BaseModel):
    """Model identifiers for the different reasoning roles."""

    reasoning: str = "OPENAI_GPT_OSS_20B"
    writing: str = "OPENAI_GPT_OSS_20B"
    backend: Literal["ollama", "lmstudio"] = "ollama"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_llm_io: bool = Field(
        True,
        description="Log every LLM request and response at DEBUG level.",
    )


class AgentTask(BaseModel):
    """Base task: a query and a list of source documents or directories.

    Subclass to add mode-specific parameters. The ``mode`` field acts as
    a discriminator so ``load_task`` can instantiate the right subclass from
    a YAML file automatically.

    When ``mode`` is ``None`` (or omitted in YAML), the orchestrator enters
    planning mode and determines the appropriate sub-tasks automatically.
    """

    mode: str | None = Field(
        None,
        description="Task mode. None means auto-plan; subclasses override with a Literal.",
    )
    query: str = Field(..., description="The natural-language query or instruction.")
    sources: list[str] = Field(
        default_factory=list,
        description="Paths to documents or directories.",
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through extra args forwarded to specific agents.",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v: Any) -> list[str]:
        """Accept a bare string as a single-item list."""
        if isinstance(v, str):
            return [v]
        return v


class RAGTask(AgentTask):
    """Query one or more documents using the chunkless RAG loop."""

    mode: Literal["rag"] = "rag"
    max_iterations: int = Field(5, ge=1, description="Maximum section-selection iterations.")
    enrich_before_rag: bool = Field(True, description="Run summarization enrichment before the RAG loop.")

    @model_validator(mode="after")
    def sources_required(self) -> RAGTask:
        if not self.sources:
            raise ValueError("'sources' must not be empty for RAG tasks")
        return self


class ExtractTask(AgentTask):
    """Extract structured data from documents according to a schema."""

    mode: Literal["extract"] = "extract"
    schema_path: Path | None = Field(
        None,
        description="Optional path to a JSON schema file. If omitted the schema is inferred from the query.",
    )
    glob: str | None = Field(None, description="Glob pattern applied when sources contain directories.")

    @model_validator(mode="after")
    def sources_required(self) -> ExtractTask:
        if not self.sources:
            raise ValueError("'sources' must not be empty for extraction tasks")
        return self


class WriteTask(AgentTask):
    """Write a new document, optionally grounded in source documents."""

    mode: Literal["write"] = "write"
    # sources are optional for writing tasks


class EditingTask(AgentTask):
    """Edit an existing document."""

    mode: Literal["edit"] = "edit"
    # sources are optional for editing tasks


class EnrichTask(AgentTask):
    """Enrich documents with summaries, keywords, or entity annotations."""

    mode: Literal["enrich"] = "enrich"
    operations: Annotated[
        list[Literal["summarize", "keywords", "entities"]], Field(description="Enrichment operations to apply.")
    ] = ["summarize"]

    @model_validator(mode="after")
    def sources_required(self) -> EnrichTask:
        if not self.sources:
            raise ValueError("'sources' must not be empty for enrichment tasks")
        return self


# Discriminated union — Pydantic selects the right subclass via the ``mode`` field.
AnyTask = Annotated[
    RAGTask | ExtractTask | WriteTask | EditingTask | EnrichTask,
    Field(discriminator="mode"),
]

_task_adapter: TypeAdapter[AnyTask] = TypeAdapter(AnyTask)


def load_task(
    path: Path,
) -> RAGTask | ExtractTask | WriteTask | EditingTask | EnrichTask | AgentTask:
    """Load and validate a task from a YAML file.

    The ``mode`` key selects the task subclass. If ``mode`` is omitted or
    ``null``, the base ``AgentTask`` is returned with ``mode=None`` and the
    orchestrator will enter planning mode.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Task YAML must be a mapping, got {type(data).__name__}")
    if not data.get("mode"):
        return AgentTask.model_validate(data)
    return _task_adapter.validate_python(data)
