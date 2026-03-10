import datetime
import logging
import threading
import time

_START_TIME = time.time()


class _LoguruStyleFormatter(logging.Formatter):
    """Formatter that mimics the loguru C++ log style.

    Example output:
        2026-03-10 09:39:51.346 (   0.001s) [main thread     ]       logging.py:42    INFO| message
    """

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.datetime.fromtimestamp(record.created)
        ms = dt.microsecond // 1000
        date_str = dt.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d}")

        uptime = record.created - _START_TIME
        uptime_str = f"({uptime:8.3f}s)"

        thread_name = threading.current_thread().name
        thread_str = f"[{thread_name:<16}]"

        file_line = f"{record.filename}:{record.lineno}"
        file_line_str = f"{file_line:>32}"

        level_str = f"{record.levelname:>5}|"

        return f"{date_str} {uptime_str} {thread_str} {file_line_str} {level_str} {record.getMessage()}"


# Centralized logger for the docling_agent package.
# Modules should import as: `from docling_agent.logging import logger`.

logger = logging.getLogger("docling_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_LoguruStyleFormatter())
    logger.addHandler(_handler)

logger.setLevel(logging.INFO)
