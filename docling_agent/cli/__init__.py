import argparse
import json
from pathlib import Path
from typing import Iterable

from mellea.backends import model_ids

from docling_core.types.doc.document import DoclingDocument

from docling_agent.agents import DoclingExtractingAgent, logger


def resolve_model_id(name: str):
    # Try to resolve a constant in mellea.backends.model_ids by name; fallback to default
    try:
        return getattr(model_ids, name)
    except Exception:
        logger.warning(f"Unknown model id '{name}', falling back to OPENAI_GPT_OSS_20B")
        return model_ids.OPENAI_GPT_OSS_20B


def gather_sources(
    paths: Iterable[Path], pattern: str | None = None
) -> list[DoclingDocument | Path]:
    files: list[DoclingDocument | Path] = []
    for p in paths:
        if p.is_dir():
            if pattern:
                files.extend([q for q in p.rglob(pattern) if q.is_file()])
            else:
                files.extend([q for q in p.rglob("*") if q.is_file()])
        elif p.is_file():
            files.append(p)
        else:
            logger.warning(f"Ignoring non-existent path: {p}")
    return files


def cmd_extract(args: argparse.Namespace) -> int:
    schema_path = Path(args.schema)
    with open(schema_path, "r", encoding="utf-8") as fp:
        schema = json.load(fp)

    sources = gather_sources([Path(s) for s in args.sources], pattern=args.glob)
    if not sources:
        logger.error("No sources found to extract from.")
        return 2

    agent = DoclingExtractingAgent(model_id=resolve_model_id(args.model), tools=[])

    task = f"""Extract data according to this schema:

```json
{json.dumps(schema, indent=2)}
```
"""

    logger.info(f"Running extraction on {len(sources)} file(s) with model {args.model}")
    doc = agent.run(task=task, sources=sources)

    # Save document if requested
    if args.output:
        outpath = Path(args.output)
        doc.save_as_html(filename=outpath)
        logger.info(f"Report written to '{outpath}'")

    # Dump results dict
    results_map = agent.last_results
    if args.results:
        rpath = Path(args.results)
        with open(rpath, "w", encoding="utf-8") as fp:
            json.dump(results_map, fp, indent=2)
        logger.info(f"Results written to '{rpath}'")
    else:
        print(json.dumps(results_map, indent=2))

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docling-agent")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract data from documents using a schema")
    e.add_argument(
        "--schema",
        required=True,
        help="Path to a JSON schema file",
    )
    e.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source paths (files or directories)",
    )
    e.add_argument(
        "--glob",
        default=None,
        help="Optional glob pattern when walking directories (e.g. '*.pdf')",
    )
    e.add_argument(
        "--model",
        default="OPENAI_GPT_OSS_20B",
        help="Model id constant name in mellea.backends.model_ids",
    )
    e.add_argument(
        "--output",
        help="Optional HTML report path for the extraction document",
    )
    e.add_argument(
        "--results",
        help="Optional JSON file path for the results dictionary; prints to stdout if omitted",
    )
    e.set_defaults(func=cmd_extract)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
