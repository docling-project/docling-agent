from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator
from typing_extensions import Self


class OutputConfig(BaseModel):
    """Configuration for output file generation.

    Specifies where to write results and in which formats. Supports multiple
    output formats simultaneously (markdown, HTML, JSON).
    """

    path: Annotated[
        Path | None,
        Field(description="Specific output file path. If None, auto-generates based on dir and timestamp."),
    ] = None
    dir: Annotated[
        Path,
        Field(description="Directory for output files when path is not specified."),
    ] = Path("./outputs")
    formats: Annotated[
        list[Literal["markdown", "html", "json"]],
        Field(description="Output formats to generate. Duplicates are automatically removed."),
    ] = ["markdown", "html", "json"]

    @model_validator(mode="after")
    def normalize_formats(self) -> Self:
        """Remove duplicate formats and validate at least one format is specified.

        Raises:
            ValueError: If formats list is empty after deduplication.
        """
        self.formats = list(dict.fromkeys(self.formats))
        if not self.formats:
            raise ValueError("'output.formats' must contain at least one format")
        return self


class ModelConfig(BaseModel):
    """Model identifiers for different agent roles.

    - reasoning: Used for planning, analysis, and decision-making
    - writing: Used for content generation and document creation
    - extraction: Used for structured data extraction (defaults to writing)
    """

    reasoning: Annotated[
        str,
        Field(description="Model identifier for reasoning tasks (planning, analysis, decision-making)."),
    ] = "OPENAI_GPT_OSS_20B"
    writing: Annotated[
        str,
        Field(description="Model identifier for writing tasks (content generation, document creation)."),
    ] = "OPENAI_GPT_OSS_20B"
    extraction: Annotated[
        str | None,
        Field(description="Model identifier for extraction tasks. Defaults to writing model if not specified."),
    ] = None

    @model_validator(mode="after")
    def default_extraction_model(self) -> Self:
        """Set extraction model to writing model if not explicitly specified."""
        if self.extraction is None:
            self.extraction = self.writing
        return self


class BackendConfig(BaseModel):
    """Configuration for LLM backend selection and connection.

    Supports multiple backend types (Mellea, Ollama, LM Studio, LiteLLM) with
    customizable connection settings and model assignments.
    """

    type: Annotated[
        Literal["ollama", "lmstudio", "litellm", "mellea"],
        Field(description="Backend type to use for LLM inference."),
    ] = "mellea"
    base_url: Annotated[
        str | None,
        Field(description="Custom base URL for the backend API. If None, uses backend default."),
    ] = None
    timeout: Annotated[
        int | None,
        Field(description="Request timeout in seconds. If None, uses backend default."),
    ] = None
    api_key_env: Annotated[
        str | None,
        Field(
            repr=False,
            description="Environment variable name containing the API key. Hidden from repr for security.",
        ),
    ] = None
    options: Annotated[
        dict[str, Any],
        Field(description="Backend-specific options passed through to the provider."),
    ] = {}
    models: Annotated[
        ModelConfig,
        Field(description="Model identifiers for different agent roles."),
    ] = ModelConfig()


class LoggingConfig(BaseModel):
    """Configuration for logging behavior.

    Controls log levels and optional detailed LLM interaction logging for debugging.
    """

    level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        Field(description="Minimum log level to output."),
    ] = "INFO"
    log_llm_io: Annotated[
        bool,
        Field(description="Log every LLM request and response at DEBUG level."),
    ] = True
    linear_chat_log_path: Annotated[
        Path | None,
        Field(description="Optional file path to append linearized chat contexts during a run."),
    ] = None


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
    backend: BackendConfig = Field(default_factory=BackendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through extra args forwarded to specific agents.",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v: Any) -> list[str]:
        """Convert single string source to list for convenience.

        Args:
            v: Either a string or list of strings.
        """
        if isinstance(v, str):
            return [v]
        return v


class RAGTask(AgentTask):
    """Retrieval-Augmented Generation task configuration.

    Queries documents using an iterative section-selection approach without
    traditional chunking. Optionally enriches documents with summaries before querying.
    """

    mode: Literal["rag"] = "rag"
    max_iterations: Annotated[
        int,
        Field(ge=1, description="Maximum section-selection iterations."),
    ] = 5
    enrich_before_rag: Annotated[
        bool,
        Field(description="Run summarization enrichment before the RAG loop."),
    ] = True

    @model_validator(mode="after")
    def sources_required(self) -> Self:
        """Validate that at least one source document is provided.

        Raises:
            ValueError: If sources list is empty.
        """
        if not self.sources:
            raise ValueError("'sources' must not be empty for RAG tasks")
        return self


class ExtractTask(AgentTask):
    """Structured data extraction task configuration.

    Extracts data from documents according to a JSON schema. Schema can be
    provided explicitly or inferred from the query.
    """

    mode: Literal["extract"] = "extract"
    schema_path: Annotated[
        Path | None,
        Field(description="Optional path to a JSON schema file. If omitted the schema is inferred from the query."),
    ] = None
    glob: Annotated[
        str | None,
        Field(description="Glob pattern applied when sources contain directories."),
    ] = None

    @model_validator(mode="after")
    def sources_required(self) -> Self:
        """Validate that at least one source document is provided.

        Raises:
            ValueError: If sources list is empty.
        """
        if not self.sources:
            raise ValueError("'sources' must not be empty for extraction tasks")
        return self


class WriteTask(AgentTask):
    """Document writing task configuration.

    Creates new documents from scratch or grounded in source materials.
    Sources are optional - can write purely from the query.
    """

    mode: Literal["write"] = "write"


class EditingTask(AgentTask):
    """Document editing task configuration.

    Modifies existing documents based on instructions. Sources are optional
    but typically include the document to be edited.
    """

    mode: Literal["edit"] = "edit"


class EnrichTask(AgentTask):
    """Document enrichment task configuration.

    Adds metadata to documents through various operations: summarization,
    keyword extraction, entity recognition, and classification.
    """

    mode: Literal["enrich"] = "enrich"
    operations: Annotated[
        list[Literal["summarize", "keywords", "entities", "classify", "classify_items"]] | None,
        Field(description="Enrichment operations to apply. If None, applies all available operations."),
    ] = None

    @model_validator(mode="after")
    def sources_required(self) -> EnrichTask:
        """Validate that at least one source document is provided.

        Raises:
            ValueError: If sources list is empty.
        """
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
