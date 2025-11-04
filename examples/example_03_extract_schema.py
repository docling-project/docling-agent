from pathlib import Path
import json

from mellea.backends import model_ids

from docling_core.types.doc.document import (
    DoclingDocument,
)

from docling_agent.agents import DoclingExtractingAgent, logger


def run_task(
    schema: dict,
    sources: list[Path],
    opath: Path,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list | None = None,
):
    agent = DoclingExtractingAgent(model_id=model_id, tools=tools or [])

    task = f"""Extract data according to this schema:

```json
{json.dumps(schema, indent=2)}
```
"""

    document = agent.run(
        task=task,
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
        "skills": "string"
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
    
    sources = Path("./examples")  # Adjust to your data root

    for _ in [
        (
            schema_01,
            sources / "curriculum_vitae",
        ),
        (
            schema_02,
            sources / "papers",
        ),
        (
            schema_03,
            sources / "invoices",
        )        
    ]:
        run_task(schema=_[0], sources=[_[1]], opath=Path("./extraction_report.html"), model_id=model_id)
    
if __name__ == "__main__":
    main()
