from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from mellea.backends import ModelIdentifier, model_ids

from docling_agent.agents import DoclingWritingAgent, logger


def simple_writing_report(task: str, out: Path, model_id: ModelIdentifier) -> None:
    tools = []

    agent = DoclingWritingAgent(model_id=model_id, tools=tools)
    document = agent.run(task=task)

    # Save the document
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename= out / Path(f"{fname}.json"))
    document.save_as_markdown(filename=out / Path(f"{fname}.md"), text_width=72)
    out_html = out / Path(f"{fname}.html")
    document.save_as_html(filename=out_html)

    logger.info(f"report written to `{out_html}`")

def advanced_writing_report(task: str, out: Path, reasoning_model: ModelIdentifier, writing_model: ModelIdentifier) -> None:
    tools = []

    # Initialize the agent with a base model id
    agent = DoclingWritingAgent(model_id=reasoning_model, tools=tools)
    # Configure specialized models for reasoning and writing
    agent.reasoning_model_id = reasoning_model
    agent.writing_model_id = writing_model

    document = agent.run(task=task)

    # Save the document
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename= out / Path(f"{fname}.json"))
    document.save_as_markdown(filename=out / Path(f"{fname}.md"), text_width=72)
    out_html = out / Path(f"{fname}.html")
    document.save_as_html(filename=out_html)

    logger.info(f"report written to `{out_html}`")

def main():
    out_dir = Path("scratch/example_01")
    os.makedirs(out_dir, exist_ok=True)
    reasoning_model_id = model_ids.OPENAI_GPT_OSS_20B
    writing_model_id = model_ids.IBM_GRANITE_4_MICRO_3B

    task = (
        "Write me a document on polymers in food-packaging. Please make sure "
        "that you have a table listing all the most common polymers and their "
        "properties, a section on biodegradability and common practices to improve "
        "strength and durability."
    )

    # simple_writing_report(task=task, out = out_dir, model_id=writing_model_id)

    advanced_writing_report(task=task, out = out_dir, reasoning_model=reasoning_model_id, writing_model=writing_model_id)


if __name__ == "__main__":
    main()
