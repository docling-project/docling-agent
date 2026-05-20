from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock

from tabulate import tabulate

from docling_agent.logging import logger

# Whether to log LLM requests/responses — toggled by the orchestrator via configure_llm_logging().
_log_llm_io: bool = False
_linear_chat_log_path: Path | None = None
_linear_chat_log_lock = Lock()


def configure_llm_logging(enabled: bool) -> None:
    """Enable or disable per-call LLM request/response logging."""
    global _log_llm_io
    _log_llm_io = enabled


def configure_linear_chat_logging(path: Path | None) -> None:
    """Enable or disable appending linearized chat contexts to a file."""
    global _linear_chat_log_path
    _linear_chat_log_path = path


def should_log_llm_io() -> bool:
    """Return whether per-call LLM request/response logging is enabled."""
    return _log_llm_io


def view_linear_context(session) -> None:
    """Log the current session context when the backend supports inspection."""
    rows = session.debug_context_rows()
    if not rows:
        return
    rendered = tabulate(rows, headers=["turn", "role", "message"])
    logger.info(f"linearized chat:\n\n {rendered}")
    if _linear_chat_log_path is None:
        return
    with _linear_chat_log_lock:
        _linear_chat_log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        block = f"\n[{stamp}] linearized chat\n{rendered}\n"
        with _linear_chat_log_path.open("a", encoding="utf-8") as fh:
            fh.write(block)
