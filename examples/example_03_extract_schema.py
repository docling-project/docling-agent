from pathlib import Path
import json

from mellea.backends import model_ids

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
    model_id = model_ids.OPENAI_GPT_OSS_20B

    schema_01 = {
        "name": "string",
        "birth year": "integer",
        "nationality": "string",
        "contact details": "string",
        "latest education": "string",
        "languages": "string",
        "skills": "string",
    }

    schema_02 = {
        "title": "string",
        "authors": "string"
    }

    schema_03 = {
        "invoice-number": "string",
        "total": "float",
        "currency": "string",
    }
    
    docdir = Path("./examples/example_03_extract")  # Adjust to your data root

    for _ in [
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
        )        
    ]:
        cdir = docdir / _[1]
        
        sources: list[Path] = []
        # Collect PDFs and PNGs recursively under each source directory
        sources.extend([p for p in cdir.rglob("*.pdf") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.png") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.jpg") if p.is_file()])
        sources.extend([p for p in cdir.rglob("*.jpeg") if p.is_file()])

        sources = sorted(sources)
        
        logger.info(f"documents [{len(sources)}]:\n\n\t" + ",\n\t".join(str(p) for p in sources))

        run_task(
            schema=_[0],
            sources=sources,
            opath=docdir / f"{_[1]}_extraction_report.html",
            model_id=model_id,
        )
    
if __name__ == "__main__":
    main()
