from __future__ import annotations

from docling_agent.backends.base import BaseBackend
from docling_agent.backends.registry import get_backend_class
from docling_agent.task_model import BackendConfig


def create_backend(config: BackendConfig) -> BaseBackend:
    """Factory function to instantiate a backend from configuration.

    Looks up the appropriate backend class based on the config type
    and initializes it with the provided settings.

    Args:
        config: Backend configuration specifying type and connection settings.

    Returns:
        Initialized backend instance ready for use.

    Raises:
        RuntimeError: If backend initialization fails, wrapping the underlying exception.
        ValueError: If the backend type is not recognized (raised by get_backend_class).
    """
    backend_cls = get_backend_class(config.type)
    try:
        return backend_cls.from_config(config)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize {config.type!r} backend: {exc}") from exc
