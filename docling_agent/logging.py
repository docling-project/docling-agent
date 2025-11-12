import logging


# Centralized logger for the docling_agent package.
# Modules should import as: `from docling_agent.logging import logger`.

logger = logging.getLogger("docling_agent")

# Configure default handler/format only if not already configured by host app
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Set a sensible default level; host apps may override
logger.setLevel(logging.INFO)

