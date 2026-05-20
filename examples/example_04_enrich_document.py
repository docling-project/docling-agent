import os
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument

from docling_agent.agents import BackendConfig, DoclingEnrichingAgent, ModelConfig, create_backend, logger


def run_task(
    ipath: Path,
    opath: Path,
    task: str,
    model_name: str = "OPENAI_GPT_OSS_20B",
    tools: list | None = None,
):
    document = DoclingDocument.load_from_json(ipath)

    agent = DoclingEnrichingAgent(
        backend=create_backend(
            BackendConfig(
                type="mellea",
                models=ModelConfig(reasoning=model_name, writing=model_name),
            )
        ),
        tools=tools or [],
    )

    document = agent.run(task=task, document=document)
    document.save_as_html(filename=opath)

    logger.info(f"enrichment report written to `{opath}`")


def main():
    out_dir = Path("scratch/example_04")
    os.makedirs(out_dir, exist_ok=True)
    model_name = "OPENAI_GPT_OSS_20B"

    # Example document to enrich (reuse the editing sample document)
    ipath = Path("examples/data/20250815_125216.json")

    tasks: list[tuple[str, str]] = [
        (
            "Summarize each paragraph, table, and section header in this document.",
            out_dir / Path(ipath.stem + "_summaries.html"),
        ),
        (
            "Find search keywords for each paragraph, table, and section header.",
            out_dir / Path(ipath.stem + "_keywords.html"),
        ),
        (
            "Detect key entities across paragraphs, tables, and sections.",
            out_dir / Path(ipath.stem + "_entities.html"),
        ),
    ]

    for task, output in tasks:
        run_task(ipath=ipath, opath=output, task=task, model_name=model_name)


if __name__ == "__main__":
    main()
