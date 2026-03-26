import json
import os
from pathlib import Path

from mellea.backends import ModelIdentifier, model_ids

from docling_agent.agents import DoclingExtractingAgent, logger


def run_task(
    schema: dict,
    sources: list[Path],
    opath: Path,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list | None = None,
):
    agent = DoclingExtractingAgent(model_id=model_id, tools=tools or [])

    document = agent.run(
        task=json.dumps(schema),
        sources=sources,
    )
    document.save_as_html(filename=opath)

    logger.info(f"report written to `{opath}`")


def main():
    doc_dir: Path = Path("examples/data")
    out_dir: Path = Path("scratch/example_03")
    os.makedirs(out_dir, exist_ok=True)
    model_id: ModelIdentifier = model_ids.OPENAI_GPT_OSS_20B

    schema_01: dict[str, str] = {
        "name": "string",
        "birth year": "integer",
        "nationality": "string",
        "contact details": "string",
        "latest education": "string",
        "languages": "string",
        "skills": "string",
    }

    schema_02: dict[str, str] = {"title": "string", "authors": "string"}

    schema_03: dict[str, str] = {
        "invoice-number": "string",
        "total": "float",
        "currency": "string",
    }

    for schema, type in [
        (
            schema_01,
            "curriculum_vitae",
        ),
        (
            schema_02,
            "papers",
        ),
        (
            schema_03,
            "invoices",
        ),
    ]:
        cdir = doc_dir / type

        sources: list[Path] = []
        # Collect PDFs and PNGs recursively under each source directory
        sources.extend([p for p in cdir.rglob("*.pdf") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.png") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.jpg") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.jpeg") if p.is_file()])

        sources = sorted(sources)

        logger.info(
            f"Extract {list(schema)} from documents [{len(sources)}]:\n\n\t" + ",\n\t".join(str(p) for p in sources)
        )

        run_task(
            schema=schema,
            sources=sources,
            opath=out_dir / f"{type}_extraction_report.html",
            model_id=model_id,
        )


if __name__ == "__main__":
    main()
