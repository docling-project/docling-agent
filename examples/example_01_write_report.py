import os
from datetime import datetime

from mellea.backends import model_ids

from docling_agent.agents import DoclingWritingAgent, logger


def simple_writing_report(task: str):

    model_id = model_ids.OPENAI_GPT_OSS_20B
    # model_id = model_ids.IBM_GRANITE_4_MICRO_3B

    # tools_config = MCPConfig()
    # tools = setup_mcp_tools(config=tools_config)
    tools = []

    agent = DoclingWritingAgent(model_id=model_id, tools=tools)
    document = agent.run(task=task)

    # Save the document
    os.makedirs("./scratch", exist_ok=True)
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename=f"./scratch/{fname}.json")
    document.save_as_markdown(filename=f"./scratch/{fname}.md", text_width=72)
    document.save_as_html(filename=f"./scratch/{fname}.html")

    logger.info(f"report written to `./scratch/{fname}.html`")

def advanced_writing_report(task: str):

    reasoning_model_id = model_ids.OPENAI_GPT_OSS_20B
    writing_model_id = model_ids.IBM_GRANITE_4_MICRO_3B

    # tools_config = MCPConfig()
    # tools = setup_mcp_tools(config=tools_config)
    tools = []

    # Initialize the agent with a base model id
    agent = DoclingWritingAgent(model_id=reasoning_model_id, tools=tools)
    # Configure specialized models for reasoning and writing
    agent.reasoning_model_id = reasoning_model_id
    agent.writing_model_id = writing_model_id
    
    document = agent.run(task=task)

    # Save the document
    os.makedirs("./scratch", exist_ok=True)
    fname = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    document.save_as_json(filename=f"./scratch/{fname}.json")
    document.save_as_markdown(filename=f"./scratch/{fname}.md", text_width=72)
    document.save_as_html(filename=f"./scratch/{fname}.html")

    logger.info(f"report written to `./scratch/{fname}.html`")    

def main():

    task = (
        "Write me a document on polymers in food-packaging. Please make sure "
        "that you have a table listing all the most common polymers and their "
        "properties, a section on biodegradability and common practices to improve "
        "strength and durability."
    )
    
    # simple_writing_report(task=task)

    advanced_writing_report(task=task)

    
    
if __name__ == "__main__":
    main()
