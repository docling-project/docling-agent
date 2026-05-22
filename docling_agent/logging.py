import datetime
import logging
import threading
import time
from contextlib import contextmanager
from enum import Enum

_START_TIME = time.time()


class LogLevel(Enum):
    """Log levels for agent operations."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class AgentLogContext:
    """Thread-local context for agent logging."""

    def __init__(self):
        self._local = threading.local()

    @property
    def agent_name(self) -> str | None:
        return getattr(self._local, "agent_name", None)

    @agent_name.setter
    def agent_name(self, value: str | None):
        self._local.agent_name = value

    @property
    def operation(self) -> str | None:
        return getattr(self._local, "operation", None)

    @operation.setter
    def operation(self, value: str | None):
        self._local.operation = value

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @depth.setter
    def depth(self, value: int):
        self._local.depth = value


_agent_context = AgentLogContext()


class _AgenticLogFormatter(logging.Formatter):
    """Simplified formatter optimized for agentic workflows.

    Example output:
        [2026-05-20 15:19:02.401]    INFO | 🚀 [WriterAgent] Starting document writing
    """

    def format(self, record: logging.LogRecord) -> str:
        # Compact timestamp with date (YYYY-MM-DD HH:MM:SS.mmm)
        dt = datetime.datetime.fromtimestamp(record.created)
        ms = dt.microsecond // 1000
        time_str = dt.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d}")

        # Level indicator
        level_str = f"{record.levelname:>7}"

        return f"[{time_str}] {level_str} | {record.getMessage()}"


# Centralized logger for the docling_agent package.
# Modules should import as: `from docling_agent.logging import logger`.

logger = logging.getLogger("docling_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_AgenticLogFormatter())
    logger.addHandler(_handler)

logger.setLevel(logging.INFO)


# ============================================================================
# Modern Agentic Logging API
# ============================================================================


def _get_indent() -> str:
    """Get indentation based on current depth."""
    return "  " * _agent_context.depth


def _format_agent_prefix() -> str:
    """Format the agent context prefix."""
    parts = []
    if _agent_context.agent_name:
        parts.append(f"[{_agent_context.agent_name}]")
    if _agent_context.operation:
        parts.append(f"({_agent_context.operation})")
    return " ".join(parts) if parts else ""


@contextmanager
def agent_context(agent_name: str):
    """Context manager to set the current agent name for logging.

    Usage:
        with agent_context("WriterAgent"):
            log_agent_start("Writing document")
            # ... agent operations ...
            log_agent_end("Document written")
    """
    old_name = _agent_context.agent_name
    _agent_context.agent_name = agent_name
    try:
        yield
    finally:
        _agent_context.agent_name = old_name


@contextmanager
def operation_context(operation_name: str):
    """Context manager to set the current operation for logging.

    Usage:
        with operation_context("outline_generation"):
            log_stage_start("Generating outline")
            # ... operation steps ...
            log_stage_end("Outline generated")
    """
    old_operation = _agent_context.operation
    old_depth = _agent_context.depth
    _agent_context.operation = operation_name
    _agent_context.depth = old_depth + 1
    try:
        yield
    finally:
        _agent_context.operation = old_operation
        _agent_context.depth = old_depth


def log_agent_start(message: str, **kwargs):
    """Log the start of an agent operation."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}🚀 {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_agent_end(message: str, **kwargs):
    """Log the end of an agent operation."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}✅ {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_stage_start(stage_name: str, **kwargs):
    """Log the start of a processing stage."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}▶️  {prefix} Stage: {stage_name}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_stage_end(stage_name: str, duration: float | None = None, **kwargs):
    """Log the end of a processing stage."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}⏹️  {prefix} Stage complete: {stage_name}"
    if duration is not None:
        full_msg += f" | duration={duration:.2f}s"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_llm_request(
    prompt: str,
    model: str | None = None,
    max_preview_length: int = 200,
    **kwargs,
):
    """Log an LLM request with structured formatting.

    Args:
        prompt: The prompt being sent to the LLM
        model: The model being used (optional)
        max_preview_length: Maximum length of prompt preview
        **kwargs: Additional metadata (temperature, max_tokens, etc.)
    """
    prefix = _format_agent_prefix()
    indent = _get_indent()

    # Truncate prompt for preview
    prompt_preview = prompt if len(prompt) <= max_preview_length else f"{prompt[:max_preview_length]}..."

    # Build metadata string
    metadata = []
    if model:
        metadata.append(f"model={model}")
    metadata.extend(f"{k}={v}" for k, v in kwargs.items())
    metadata_str = " | ".join(metadata) if metadata else ""

    logger.info(f"{indent}📤 {prefix} LLM Request{' | ' + metadata_str if metadata_str else ''}")
    logger.debug(f"{indent}   Prompt: {prompt_preview}")


def log_llm_response(
    response: str,
    model: str | None = None,
    max_preview_length: int = 200,
    **kwargs,
):
    """Log an LLM response with structured formatting.

    Args:
        response: The response from the LLM
        model: The model that generated the response (optional)
        max_preview_length: Maximum length of response preview
        **kwargs: Additional metadata (tokens, duration, etc.)
    """
    prefix = _format_agent_prefix()
    indent = _get_indent()

    # Truncate response for preview
    response_preview = response if len(response) <= max_preview_length else f"{response[:max_preview_length]}..."

    # Build metadata string
    metadata = []
    if model:
        metadata.append(f"model={model}")
    metadata.extend(f"{k}={v}" for k, v in kwargs.items())
    metadata_str = " | ".join(metadata) if metadata else ""

    logger.info(f"{indent}📥 {prefix} LLM Response{' | ' + metadata_str if metadata_str else ''}")
    logger.debug(f"{indent}   Response: {response_preview}")


def log_llm_interaction(
    turn: int,
    role: str,
    content_preview: str,
    **kwargs,
):
    """Log a single turn in a multi-turn LLM conversation.

    Args:
        turn: The turn number in the conversation
        role: The role (user, assistant, system)
        content_preview: Preview of the message content
        **kwargs: Additional metadata
    """
    indent = _get_indent()

    role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(role.lower(), "💬")

    metadata = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}  {role_emoji} Turn {turn} [{role}]: {content_preview}"
    if metadata:
        full_msg += f" | {metadata}"

    logger.debug(full_msg)


def log_validation_attempt(attempt: int, max_attempts: int, success: bool, reason: str | None = None):
    """Log a validation attempt for structured output."""
    indent = _get_indent()

    status = "✓" if success else "✗"
    msg = f"{indent}  {status} Validation attempt {attempt}/{max_attempts}"
    if not success and reason:
        msg += f" | reason: {reason}"

    logger.debug(msg)


def log_info(message: str, **kwargs):
    """Log an informational message with agent context."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}ℹ️  {prefix} {message}"  # noqa: RUF001
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_warning(message: str, **kwargs):
    """Log a warning message with agent context."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}⚠️  {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.warning(full_msg)


def log_error(message: str, exception: Exception | None = None, **kwargs):
    """Log an error message with agent context."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}❌ {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    if exception:
        full_msg += f" | exception: {type(exception).__name__}: {exception!s}"
    logger.error(full_msg)


def log_success(message: str, **kwargs):
    """Log a success message with agent context."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}✨ {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.info(full_msg)


def log_debug(message: str, **kwargs):
    """Log a debug message with agent context."""
    prefix = _format_agent_prefix()
    indent = _get_indent()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_msg = f"{indent}🔍 {prefix} {message}"
    if extra_info:
        full_msg += f" | {extra_info}"
    logger.debug(full_msg)


@contextmanager
def timed_operation(operation_name: str):
    """Context manager to time an operation and log the duration.

    Usage:
        with timed_operation("document_conversion"):
            # ... operation ...
    """
    start_time = time.time()
    log_stage_start(operation_name)
    try:
        yield
    finally:
        duration = time.time() - start_time
        log_stage_end(operation_name, duration=duration)
