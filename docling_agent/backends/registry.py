"""Backend registry for LLM provider implementations.

This module maintains a central registry of available backend implementations
and provides lookup functionality for the factory pattern.
"""

from __future__ import annotations

from docling_agent.backends.base import BaseBackend
from docling_agent.backends.litellm_backend import LiteLLMBackend
from docling_agent.backends.lmstudio_backend import LMStudioBackend
from docling_agent.backends.mellea_backend import MelleaBackend
from docling_agent.backends.ollama_backend import OllamaBackend

BACKEND_REGISTRY: dict[str, type[BaseBackend]] = {
    "mellea": MelleaBackend,
    "ollama": OllamaBackend,
    "lmstudio": LMStudioBackend,
    "litellm": LiteLLMBackend,
}
"""Registry mapping backend type names to their implementation classes.

To add a new backend, import the class and add an entry here.
"""


def get_backend_class(name: str) -> type[BaseBackend]:
    """Look up a backend implementation class by its configuration name.

    Args:
        name: Backend type identifier (e.g., "mellea", "ollama", "lmstudio", "litellm").

    Returns:
        The backend class corresponding to the given name.

    Raises:
        ValueError: If the backend name is not found in the registry.
            The error message includes a list of supported backends.
    """
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unknown backend type {name!r}. Supported backends: {supported}") from exc
