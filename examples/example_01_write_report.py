from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from docling_agent.agents import BackendConfig, DoclingWritingAgent, ModelConfig, create_backend, logger


def simple_writing_report(task: str, out: Path, model_name: str) -> None:
    tools = []

    agent = DoclingWritingAgent(
        backend=create_backend(
            BackendConfig(
                type="mellea",
                models=ModelConfig(reasoning=model_name, writing=model_name),
            )
        ),
        tools=tools,
    )
    document = agent.run(task=task)

    # Save the document
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename=out / Path(f"{fname}.json"))
    document.save_as_markdown(filename=out / Path(f"{fname}.md"), text_width=72)
    out_html = out / Path(f"{fname}.html")
    document.save_as_html(filename=out_html)

    logger.info(f"report written to `{out_html}`")


def advanced_writing_report(task: str, out: Path, reasoning_model: str, writing_model: str) -> None:
    tools = []

    agent = DoclingWritingAgent(
        backend=create_backend(
            BackendConfig(
                type="mellea",
                models=ModelConfig(reasoning=reasoning_model, writing=writing_model),
            )
        ),
        tools=tools,
    )

    document = agent.run(task=task)

    # Save the document
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename=out / Path(f"{fname}.json"))
    document.save_as_markdown(filename=out / Path(f"{fname}.md"), text_width=72)
    out_html = out / Path(f"{fname}.html")
    document.save_as_html(filename=out_html)

    logger.info(f"report written to `{out_html}`")


def main():
    out_dir = Path("scratch/example_01")
    os.makedirs(out_dir, exist_ok=True)
    reasoning_model_name = "OPENAI_GPT_OSS_20B"
    writing_model_name = "IBM_GRANITE_4_MICRO_3B"

    task = (
        "Write me a document on polymers in food-packaging. Please make sure "
        "that you have a table listing all the most common polymers and their "
        "properties, a section on biodegradability and common practices to improve "
        "strength and durability."
    )

    # simple_writing_report(task=task, out=out_dir, model_name=writing_model_name)

    advanced_writing_report(
        task=task,
        out=out_dir,
        reasoning_model=reasoning_model_name,
        writing_model=writing_model_name,
    )


if __name__ == "__main__":
    main()
