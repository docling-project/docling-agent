import os
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from mellea.backends import model_ids

from docling_agent.agents import DoclingRAGAgent, logger


def run_task(
    ipath: Path,
    opath: Path,
    query: str,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list | None = None,
    max_iterations: int = 5,
    verbose: bool = True,
):
    """Run a RAG query on a document and save the answer.

    Args:
        ipath: Path to the input DoclingDocument JSON file
        opath: Path to save the output HTML file
        query: The question to answer from the document
        model_id: The model identifier to use
        tools: Optional list of tools for the agent
        max_iterations: Maximum number of RAG iterations
        verbose: Whether to print detailed progress information
    """
    document = DoclingDocument.load_from_json(ipath)

    agent = DoclingRAGAgent(
        model_id=model_id,
        tools=tools or [],
        max_iterations=max_iterations,
        verbose=verbose,
    )

    # Run the RAG agent with the query
    answer_doc = agent.run(task=query, document=document)
    answer_doc.save_as_html(filename=opath)

    logger.info(f"RAG answer written to `{opath}`")


def main():
    out_dir = Path("scratch/example_05")
    os.makedirs(out_dir, exist_ok=True)
    model_id = model_ids.OPENAI_GPT_OSS_20B

    ipath = Path("tests/data/2408.09869v5-hierarchical-with-summaries.json")

    # Define various queries to test the RAG agent
    queries: list[tuple[str, str]] = [
        (
            "What are the main AI models used in Docling?",
            out_dir / "2408.09869v5_qa_models.html",
        ),
        (
            "Which open-source license does Docling have?",
            out_dir / "2408.09869v5_qa_license.html",
        ),
        (
            "What is the TTS with an Apple M3 and the native backend?",
            out_dir / "2408.09869v5_qa_tts.html",
        ),
        (
            "Is the optimization of CPU processes in the roadmap?",
            out_dir / "2408.09869v5_qa_roadmap.html",
        ),
    ]

    for query, output in queries:
        logger.info(f"\n{'=' * 80}\nQuery: {query}\n{'=' * 80}")
        run_task(
            ipath=ipath,
            opath=output,
            query=query,
            model_id=model_id,
            max_iterations=5,
            verbose=True,
        )


if __name__ == "__main__":
    main()
