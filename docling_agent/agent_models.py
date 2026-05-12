from __future__ import annotations

from tabulate import tabulate

from docling_agent.logging import logger

# Whether to log LLM requests/responses — toggled by the orchestrator via configure_llm_logging().
_log_llm_io: bool = False


def configure_llm_logging(enabled: bool) -> None:
    """Enable or disable per-call LLM request/response logging."""
    global _log_llm_io
    _log_llm_io = enabled


def should_log_llm_io() -> bool:
    """Return whether per-call LLM request/response logging is enabled."""
    return _log_llm_io


def view_linear_context(session) -> None:
    """Log the current session context when the backend supports inspection."""
    rows = session.debug_context_rows()
    if not rows:
        return
    logger.info(f"linearized chat:\n\n {tabulate(rows, headers=['turn', 'role', 'message'])}")
