from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import typer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import DoclingDocument

from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.agent_models import configure_linear_chat_logging, configure_llm_logging
from docling_agent.backends import create_backend
from docling_agent.logging import logger
from docling_agent.task_model import AddTask, AgentTask, ClearTask, ListTask, ViewTask, load_task

app = typer.Typer(name="docling-agent", add_completion=False, pretty_exceptions_show_locals=False)


_TASK_TEMPLATE = """\
# docling-agent task file
# Run with: uv run docling-agent --task <this-file>

# Required: the natural-language query or instruction.
query: "Your query here"

# Optional: document library project id. Defaults to "default".
project_id: default

# Required for rag / extract / enrich: paths to documents or directories.
sources:
  - path/to/document.pdf
  # - path/to/directory/

# Task mode: add | list | view | clear | rag | extract | write | edit | enrich  (omit to auto-plan)
# mode: rag

# --- Library options (mode: add | list | view | clear) ------------------------
# mode: add
# sources: [path/to/document.pdf]
# conversion: standard  # fast | standard | expensive
#
# mode: list
# postgres_filter: "project_id = 'default'"  # optional; omit to list all projects
# limit: 100
#
# mode: view
# postgres_filter: "project_id = 'default'"  # required
# limit: 100
#
# mode: clear
# project_id: default  # clear this project
# all_projects: false  # set true to clear every project

# --- RAG options (mode: rag) -------------------------------------------------
# max_iterations: 5       # maximum section-selection iterations
# enrich_before_rag: true # run summarization enrichment before querying

# --- Extract options (mode: extract) -----------------------------------------
# schema_path: schema.json  # optional JSON schema; inferred from query if omitted
# glob: "*.pdf"             # glob pattern applied when sources contain directories

# --- Enrich options (mode: enrich) -------------------------------------------
# operations:
#   - summarize   # attach 2-3 sentence summaries to each document node
#   - keywords    # extract keywords per item
#   - entities    # detect key entities per item
#   - classify    # classify pictures and attach chart/code metadata when possible

# Output configuration --------------------------------------------------------
# output:
#   dir: ./outputs                    # default output directory
#   path: result                      # optional explicit output path or basename
#   formats: [markdown, html, json]   # markdown | html | json

# Backend configuration -------------------------------------------------------
# backend:
#   type: mellea     # mellea | ollama | lmstudio | litellm | llama-server
#   base_url:
#   timeout:
#   api_key_env:
#   models:
#     reasoning: OPENAI_GPT_OSS_20B
#     writing: OPENAI_GPT_OSS_20B
#     extraction: OPENAI_GPT_OSS_20B

# Logging configuration -------------------------------------------------------
# logging:
#   level: INFO        # DEBUG | INFO | WARNING | ERROR
#   log_llm_io: true   # log every LLM request and response at DEBUG level
#   linear_chat_log_path: ./outputs/linear_chats.log
"""


@app.command()
def main(
    task: Path = typer.Option(..., "--task", "-t", help="Path to task YAML file."),
    create_task: bool = typer.Option(
        False,
        "--create-task",
        help="Write a template task YAML to --task path and exit.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override both reasoning and writing model id."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Override output path from the task file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run a docling-agent task defined in a YAML file.

    Example usage:

        uv run docling-agent --create-task -t task.yaml
        uv run docling-agent --task task.yaml
        uv run docling-agent --task task.yaml --model OPENAI_GPT_OSS_20B --verbose
    """
    if create_task:
        if task.exists():
            typer.echo(f"File already exists: {task}. Aborting to avoid overwrite.", err=True)
            raise typer.Exit(code=1)
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(_TASK_TEMPLATE, encoding="utf-8")
        typer.echo(f"Template written to {task}")
        raise typer.Exit()

    if not task.exists():
        logger.error(f"Task file not found: {task}")
        raise typer.Exit(code=1)

    agent_task = load_task(task)

    # Apply logging config from the task file (CLI --verbose overrides the level to DEBUG)
    log_level = logging.DEBUG if verbose else getattr(logging, agent_task.logging.level, logging.INFO)
    logger.setLevel(log_level)
    configure_llm_logging(agent_task.logging.log_llm_io)
    configure_linear_chat_logging(agent_task.logging.linear_chat_log_path)

    # CLI overrides
    if model:
        agent_task.backend.models.reasoning = model
        agent_task.backend.models.writing = model
    if output:
        agent_task.output.path = output

    logger.info(f"Task loaded: mode={agent_task.mode}, query={agent_task.query!r}")

    orchestrator = DoclingOrchestratorAgent(
        backend=create_backend(agent_task.backend),
        tools=[],
    )

    trace_path = agent_task.logging.trace_path
    if trace_path is not None:
        trace = orchestrator.run_task_with_trace(agent_task)
        trace.save(trace_path)
        logger.info(f"Agent trace exported to {trace_path}")
        result = cast(DoclingDocument, trace.output)
    else:
        result = orchestrator.run_task(agent_task)

    _write_output(result, agent_task, task)


def _write_output(doc, task: AgentTask, task_path: Path) -> None:
    base_path = _resolve_output_base_path(task.output, task_path)
    written_paths: list[Path] = []
    errors: list[str] = []

    for fmt in task.output.formats:
        path = _path_for_format(base_path, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if fmt == "html":
                doc.save_as_html(filename=path)
            elif fmt == "json":
                path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            else:
                path.write_text(MarkdownDocSerializer(doc=doc).serialize().text, encoding="utf-8")
            written_paths.append(path)
        except Exception as exc:
            message = f"Skipping output format {fmt!r} for {path}: {exc}"
            logger.error(message)
            errors.append(message)

    if written_paths:
        logger.info("Output written to: " + ", ".join(str(path) for path in written_paths))
    if errors and written_paths:
        logger.warning("Some output formats were skipped due to serialization errors.")
    if not written_paths:
        raise RuntimeError("All requested output formats failed to serialize.")


def _resolve_output_base_path(output, task_path: Path) -> Path:
    if output.path is not None:
        return output.path

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    return output.dir / f"{task_path.stem}_{timestamp}"


def _path_for_format(base_path: Path, fmt: str) -> Path:
    suffix_map = {
        "markdown": ".md",
        "html": ".html",
        "json": ".json",
    }
    suffix = suffix_map[fmt]
    if base_path.suffix == suffix:
        return base_path
    if base_path.suffix:
        return base_path.with_suffix(suffix)
    return base_path.parent / f"{base_path.name}{suffix}"


def _print_doc(doc) -> None:
    typer.echo(MarkdownDocSerializer(doc=doc).serialize().text)


def _run_shortcut_task(task: AgentTask, *, library_path: Path | None = None) -> None:
    orchestrator = DoclingOrchestratorAgent(
        backend=create_backend(task.backend),
        tools=[],
        library_path=library_path,
    )
    _print_doc(orchestrator.run_task(task))


def _add_sources(
    sources: list[Path] = typer.Argument(..., help="Source files or directories to add to the library."),
    project_id: str = typer.Option("default", "--project-id", "-p", help="Project id for the added documents."),
    glob: str | None = typer.Option(None, "--glob", "-g", help="Glob used when a source is a directory."),
    conversion: str = typer.Option(
        "standard",
        "--conversion",
        "-c",
        help="Conversion preset: fast, standard, or expensive.",
    ),
    library_path: Path | None = typer.Option(None, "--library-path", help="Override the document library path."),
) -> None:
    _run_shortcut_task(
        AddTask(
            project_id=project_id,
            sources=[str(source) for source in sources],
            glob=glob,
            conversion=conversion,
        ),
        library_path=library_path,
    )


def _list_sources(
    postgres_filter: str | None = typer.Option(
        None,
        "--postgres-filter",
        "-f",
        help="Optional PostgreSQL WHERE predicate for docling_library_entries.",
    ),
    limit: int = typer.Option(100, "--limit", "-n", min=1, help="Maximum number of entries to list."),
    project_id: str = typer.Option("default", "--project-id", "-p", help="Default project context."),
    library_path: Path | None = typer.Option(None, "--library-path", help="Override the document library path."),
) -> None:
    _run_shortcut_task(
        ListTask(
            project_id=project_id,
            postgres_filter=postgres_filter,
            limit=limit,
        ),
        library_path=library_path,
    )


def _view_sources(
    postgres_filter: str = typer.Option(
        ...,
        "--postgres-filter",
        "-f",
        help="Required PostgreSQL WHERE predicate for docling_library_entries.",
    ),
    limit: int = typer.Option(100, "--limit", "-n", min=1, help="Maximum number of entries to view."),
    project_id: str = typer.Option("default", "--project-id", "-p", help="Default project context."),
    library_path: Path | None = typer.Option(None, "--library-path", help="Override the document library path."),
) -> None:
    _run_shortcut_task(
        ViewTask(
            project_id=project_id,
            postgres_filter=postgres_filter,
            limit=limit,
        ),
        library_path=library_path,
    )


def _clear_sources(
    project_id: str = typer.Option("default", "--project-id", "-p", help="Project id to clear."),
    all_projects: bool = typer.Option(False, "--all", help="Clear all projects in the library."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm destructive library clearing."),
    library_path: Path | None = typer.Option(None, "--library-path", help="Override the document library path."),
) -> None:
    if not yes:
        scope = "all projects" if all_projects else f"project {project_id!r}"
        typer.echo(f"Refusing to clear {scope} without --yes.", err=True)
        raise typer.Exit(code=1)
    _run_shortcut_task(
        ClearTask(
            project_id=project_id,
            all_projects=all_projects,
        ),
        library_path=library_path,
    )


def add_sources_cli() -> None:
    typer.run(_add_sources)


def list_sources_cli() -> None:
    typer.run(_list_sources)


def view_sources_cli() -> None:
    typer.run(_view_sources)


def clear_sources_cli() -> None:
    typer.run(_clear_sources)
