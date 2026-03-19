import os
from pathlib import Path

from docling_core.types.doc.document import (
    DoclingDocument,
)
from mellea.backends import model_ids

from docling_agent.agents import DoclingEditingAgent, logger


def run_task(
    ipath: Path,
    opath: Path,
    task: str,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list = [],
):
    document = DoclingDocument.load_from_json(ipath)

    agent = DoclingEditingAgent(model_id=model_id, tools=tools)

    document = agent.run(task=task, document=document)
    document.save_as_html(filename=opath)

    logger.info(f"report written to `{opath}`")


def main():
    out_dir = Path("scratch/example_02")
    os.makedirs(out_dir, exist_ok=True)
    ipath = Path("examples/data/20250815_125216.json")
    model_id = model_ids.OPENAI_GPT_OSS_20B

    for task, output in [
        (
            "Put the polymer abbreviations in a separate column in the first table.",
            out_dir / Path(ipath.stem + "_updated_table.html")
        ),
        ("Make the title longer!",
            out_dir / Path(ipath.stem + "_updated_title.html")
        ),
        (
            "Ensure that the section headings have the correct level!",
            out_dir / Path(ipath.stem + "_updated_headings.html")
        ),
        (
            "Expand the Introduction to three paragraphs.",
            out_dir / Path(ipath.stem + "_updated_introduction.html")
        ),
    ]:
        run_task(ipath=ipath, opath=output, task=task, model_id=model_id)

if __name__ == "__main__":
    main()
