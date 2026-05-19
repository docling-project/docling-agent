from __future__ import annotations

from docling_agent.backends.base import BaseBackend
from docling_agent.backends.registry import get_backend_class
from docling_agent.task_model import BackendConfig


def create_backend(config: BackendConfig) -> BaseBackend:
    """Instantiate the configured backend."""
    backend_cls = get_backend_class(config.type)
    try:
        return backend_cls.from_config(config)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize {config.type!r} backend: {exc}") from exc
