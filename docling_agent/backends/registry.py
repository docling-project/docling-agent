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


def get_backend_class(name: str) -> type[BaseBackend]:
    """Resolve a backend implementation class from its config name."""
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unknown backend type {name!r}. Supported backends: {supported}") from exc
