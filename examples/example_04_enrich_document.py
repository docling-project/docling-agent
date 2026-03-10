import os
from pathlib import Path

from mellea.backends import model_ids

from docling_core.types.doc.document import DoclingDocument

from docling_agent.agents import DoclingEnrichingAgent, logger


def run_task(
    ipath: Path,
    task: str,
    suffix: str,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list | None = None,
):
    document = DoclingDocument.load_from_json(ipath)

    agent = DoclingEnrichingAgent(model_id=model_id, tools=tools or [])

    document = agent.run(
        task=task,
        document=document,
    )

    os.makedirs("./scratch", exist_ok=True)
    opath = Path("./scratch") / f"{ipath.stem}{suffix}.html"
    document.save_as_html(filename=opath)
    logger.info(f"enrichment report written to `{opath}`")


def main():
    model_id = model_ids.OPENAI_GPT_OSS_20B

    # Example document to enrich (reuse the editing sample document)
    ipath = Path("./examples/example_02_edit_resources/20250815_125216.json")

    tasks: list[tuple[str, str]] = [
        (
            "Summarize each paragraph, table, and section header in this document.",
            "_summaries",
        ),
        (
            "Find search keywords for each paragraph, table, and section header.",
            "_keywords",
        ),
        (
            "Detect key entities across paragraphs, tables, and sections.",
            "_entities",
        ),
        (
            "Classify items by language and function (e.g., title, abstract, claim, reference).",
            "_classifications",
        ),
    ]

    for task, suffix in tasks:
        run_task(ipath=ipath, task=task, suffix=suffix, model_id=model_id)


if __name__ == "__main__":
    main()

