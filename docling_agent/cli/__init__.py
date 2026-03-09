from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from docling_agent.agent_models import configure_llm_logging
from docling_agent.logging import logger  # type: ignore[import-untyped]
from docling_agent.task_model import AgentTask, load_task

app = typer.Typer(name="docling-agent", add_completion=False)


_TASK_TEMPLATE = """\
# docling-agent task file
# Run with: uv run docling-agent --task <this-file>

# Required: the natural-language query or instruction.
query: "Your query here"

# Required for rag / extract / enrich: paths to documents or directories.
sources:
  - path/to/document.pdf
  # - path/to/directory/

# Task mode: rag | extract | write | enrich  (default: rag)
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

# Output configuration --------------------------------------------------------
# output:
#   path: result.md   # omit to print to stdout
#   format: markdown  # markdown | html | json

# Model configuration ---------------------------------------------------------
# models:
#   reasoning: OPENAI_GPT_OSS_20B
#   writing: OPENAI_GPT_OSS_20B
#   backend: ollama  # ollama | lmstudio

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
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override both reasoning and writing model id."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Override output path from the task file."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Run a docling-agent task defined in a YAML file.

    Example usage:

        uv run docling-agent --create-task -t task.yaml
        uv run docling-agent --task task.yaml
        uv run docling-agent --task task.yaml --model OPENAI_GPT_OSS_20B --verbose
    """
    if create_task:
        if task.exists():
            typer.echo(
                f"File already exists: {task}. Aborting to avoid overwrite.", err=True
            )
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
    log_level = (
        logging.DEBUG
        if verbose
        else getattr(logging, agent_task.logging.level, logging.INFO)
    )
    logger.setLevel(log_level)
    configure_llm_logging(agent_task.logging.log_llm_io)

    # CLI overrides
    if model:
        agent_task.models.reasoning = model
        agent_task.models.writing = model
    if output:
        agent_task.output.path = output

    logger.info(f"Task loaded: mode={agent_task.mode}, query={agent_task.query!r}")

    # Deferred import: orchestrator is implemented in Feature 04.
    from mellea.backends import model_ids

    from docling_agent.agent.orchestrator import DoclingOrchestratorAgent

    def _resolve_model_id(name: str):
        resolved = getattr(model_ids, name, None)
        if resolved is None:
            logger.warning(
                f"Unknown model id '{name}', falling back to OPENAI_GPT_OSS_20B"
            )
            return model_ids.OPENAI_GPT_OSS_20B
        return resolved

    orchestrator = DoclingOrchestratorAgent(
        model_id=_resolve_model_id(agent_task.models.reasoning),
        writing_model_id=_resolve_model_id(agent_task.models.writing),
        tools=[],
    )
    result = orchestrator.run_task(agent_task)

    if agent_task.output.path:
        _write_output(result, agent_task)
    else:
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        print(MarkdownDocSerializer(doc=result).serialize().text)


def _write_output(doc, task: AgentTask) -> None:
    path = task.output.path
    if path is None:
        return
    fmt = task.output.format
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        doc.save_as_html(filename=path)
    elif fmt == "json":
        path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    else:
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        path.write_text(
            MarkdownDocSerializer(doc=doc).serialize().text, encoding="utf-8"
        )
    logger.info(f"Output written to {path}")
