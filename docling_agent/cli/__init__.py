from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import typer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

from docling_agent.agent.orchestrator import DoclingOrchestratorAgent
from docling_agent.agent_models import configure_llm_logging
from docling_agent.backends import create_backend
from docling_agent.logging import logger
from docling_agent.task_model import AgentTask, load_task

app = typer.Typer(name="docling-agent", add_completion=False, pretty_exceptions_show_locals=False)


_TASK_TEMPLATE = """\
# docling-agent task file
# Run with: uv run docling-agent --task <this-file>

# Required: the natural-language query or instruction.
query: "Your query here"

# Required for rag / extract / enrich: paths to documents or directories.
sources:
  - path/to/document.pdf
  # - path/to/directory/

# Task mode: rag | extract | write | edit | enrich  (omit to auto-plan)
# mode: rag

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
#   type: mellea     # mellea | ollama | lmstudio | litellm
#   base_url:
#   timeout:
#   api_key_env:
#   models:
#     reasoning: OPENAI_GPT_OSS_20B
#     writing: OPENAI_GPT_OSS_20B

# Logging configuration -------------------------------------------------------
# logging:
#   level: INFO        # DEBUG | INFO | WARNING | ERROR
#   log_llm_io: true   # log every LLM request and response at DEBUG level
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
