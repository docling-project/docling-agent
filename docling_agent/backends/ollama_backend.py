from __future__ import annotations

from typing import Any

import httpx
from mellea.stdlib.requirements import Requirement

from docling_agent.agent_models import should_log_llm_io
from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.logging import logger
from docling_agent.task_model import BackendConfig


class OllamaSession(BaseSession):
    """Stateful direct Ollama session."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        base_url: str,
        timeout: float,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.options = options or {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def instruct(
        self,
        prompt: str,
        *,
        requirements: list[Requirement] | None = None,
        retry_budget: int = 1,
    ) -> str:
        _ = requirements
        attempts = max(1, retry_budget)
        user_message = {"role": "user", "content": prompt}
        self._messages.append(user_message)

        if should_log_llm_io():
            logger.debug(f"[LLM REQUEST]\n{prompt}")

        last_error: Exception | None = None
        try:
            for _attempt in range(attempts):
                try:
                    response = self._client.post(
                        "/api/chat",
                        json={
                            "model": self.model,
                            "messages": [message.copy() for message in self._messages],
                            "stream": False,
                            **self.options,
                        },
                    )
                    response.raise_for_status()
                    text = self._extract_text(response.json())
                    if not text.strip():
                        raise ValueError("Ollama returned an empty response.")
                    self._messages.append({"role": "assistant", "content": text})
                    if should_log_llm_io():
                        logger.debug(f"[LLM RESPONSE]\n{text}")
                    return text
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError("Ollama request failed without an explicit error.")
        except Exception:
            self._messages.pop()
            raise

    def debug_context_rows(self) -> list[tuple[int, str, str]] | None:
        rows: list[tuple[int, str, str]] = []
        for idx, message in enumerate(self._messages):
            content = message["content"]
            if len(content) > 64:
                content = f"{content[0:32]} ... {content[-32:]}"
            rows.append((idx, message["role"], content))
        return rows

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response is missing the 'message' object.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response is missing string message content.")
        return content


class OllamaBackend(BaseBackend):
    """Direct Ollama backend."""

    backend_type = "ollama"

    def __init__(self, *, config: BackendConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.timeout = config.timeout or 120
        self.options = config.options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> OllamaBackend:
        return cls(config=config)

    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        return OllamaSession(
            model=model,
            system_prompt=system_prompt,
            base_url=self.base_url,
            timeout=float(self.timeout),
            options=self.options,
        )
