from __future__ import annotations

import os
from typing import Any

import httpx
from mellea.stdlib.requirements import Requirement

from docling_agent.agent_models import should_log_llm_io
from docling_agent.backends.base import BaseBackend, BaseSession
from docling_agent.logging import logger
from docling_agent.task_model import BackendConfig


class OpenAICompatibleSession(BaseSession):
    """Stateful direct session for OpenAI-compatible backends."""

    def __init__(
        self,
        *,
        backend_type: str,
        model: str,
        system_prompt: str | None = None,
        base_url: str,
        timeout: float,
        api_key_env: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.backend_type = backend_type
        self.model = model
        self.options = options or {}
        headers: dict[str, str] = {}
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)
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
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [message.copy() for message in self._messages],
                            **self.options,
                        },
                    )
                    response.raise_for_status()
                    text = self._extract_text(response.json())
                    if not text.strip():
                        raise ValueError(f"{self.backend_type} returned an empty response.")
                    self._messages.append({"role": "assistant", "content": text})
                    if should_log_llm_io():
                        logger.debug(f"[LLM RESPONSE]\n{text}")
                    return text
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError(f"{self.backend_type} request failed without an explicit error.")
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
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response is missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("OpenAI-compatible response choice is not a JSON object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response is missing message content.")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        raise ValueError("OpenAI-compatible response content is not supported.")


class OpenAICompatibleBackend(BaseBackend):
    """Shared config holder for OpenAI-compatible direct backends."""

    backend_type = "openai-compatible"

    def __init__(
        self,
        *,
        config: BackendConfig,
    ) -> None:
        self.config = config
        self.base_url = config.base_url or "http://localhost:4000/v1"
        self.timeout = config.timeout or 120
        self.api_key_env = config.api_key_env
        self.options = config.options or {}

    @classmethod
    def from_config(cls, config: BackendConfig) -> OpenAICompatibleBackend:
        return cls(config=config)

    def create_session(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> BaseSession:
        return OpenAICompatibleSession(
            backend_type=self.backend_type,
            model=model,
            system_prompt=system_prompt,
            base_url=self.base_url,
            timeout=float(self.timeout),
            api_key_env=self.api_key_env,
            options=self.options,
        )
