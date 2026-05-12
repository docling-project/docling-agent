from __future__ import annotations

from typing import Any

import pytest

from docling_agent.backends.litellm_backend import LiteLLMBackend
from docling_agent.backends.lmstudio_backend import LMStudioBackend
from docling_agent.backends.ollama_backend import OllamaBackend
from docling_agent.task_model import BackendConfig, ModelConfig


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        headers: dict[str, str] | None = None,
        responses: list[dict[str, Any]] | None = None,
        sink: list[dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self._responses = responses or []
        self._sink = sink if sink is not None else []

    def post(self, path: str, json: dict[str, Any]) -> _FakeResponse:
        self._sink.append({"path": path, "json": json, "headers": self.headers})
        if not self._responses:
            raise RuntimeError("No fake responses configured")
        return _FakeResponse(self._responses.pop(0))


def test_ollama_session_tracks_history(monkeypatch: pytest.MonkeyPatch):
    sink: list[dict[str, Any]] = []
    responses = [
        {"message": {"content": "First answer"}},
        {"message": {"content": "Second answer"}},
    ]

    def _fake_client_factory(*, base_url: str, timeout: float):
        return _FakeClient(base_url=base_url, timeout=timeout, responses=responses, sink=sink)

    monkeypatch.setattr("docling_agent.backends.ollama_backend.httpx.Client", _fake_client_factory)

    backend = OllamaBackend(
        config=BackendConfig(
            type="ollama",
            base_url="http://localhost:11434",
            models=ModelConfig(reasoning="qwen3:8b", writing="qwen3:8b"),
        )
    )
    session = backend.create_session(model="qwen3:8b", system_prompt="You are helpful.")

    assert session.instruct("hello") == "First answer"
    assert session.instruct("follow up") == "Second answer"

    assert sink[0]["path"] == "/api/chat"
    assert sink[0]["json"]["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]
    assert sink[1]["json"]["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "follow up"},
    ]


def test_lmstudio_session_tracks_history(monkeypatch: pytest.MonkeyPatch):
    sink: list[dict[str, Any]] = []
    responses = [
        {"choices": [{"message": {"content": "First completion"}}]},
        {"choices": [{"message": {"content": "Second completion"}}]},
    ]

    def _fake_client_factory(*, base_url: str, timeout: float, headers: dict[str, str] | None = None):
        return _FakeClient(base_url=base_url, timeout=timeout, headers=headers, responses=responses, sink=sink)

    monkeypatch.setattr("docling_agent.backends.openai_compatible.httpx.Client", _fake_client_factory)

    backend = LMStudioBackend(
        config=BackendConfig(
            type="lmstudio",
            base_url="http://localhost:1234/v1",
            models=ModelConfig(
                reasoning="granite-3.3-8b-instruct",
                writing="granite-3.3-8b-instruct",
            ),
        )
    )
    session = backend.create_session(model="granite-3.3-8b-instruct", system_prompt="You are helpful.")

    assert session.instruct("hello") == "First completion"
    assert session.instruct("follow up") == "Second completion"

    assert sink[0]["path"] == "/chat/completions"
    assert sink[0]["json"]["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]
    assert sink[1]["json"]["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "First completion"},
        {"role": "user", "content": "follow up"},
    ]


def test_litellm_session_uses_api_key_env(monkeypatch: pytest.MonkeyPatch):
    sink: list[dict[str, Any]] = []
    responses = [{"choices": [{"message": {"content": "LiteLLM answer"}}]}]
    monkeypatch.setenv("LITELLM_API_KEY", "secret-token")

    def _fake_client_factory(*, base_url: str, timeout: float, headers: dict[str, str] | None = None):
        return _FakeClient(base_url=base_url, timeout=timeout, headers=headers, responses=responses, sink=sink)

    monkeypatch.setattr("docling_agent.backends.openai_compatible.httpx.Client", _fake_client_factory)

    backend = LiteLLMBackend(
        config=BackendConfig(
            type="litellm",
            base_url="http://localhost:4000/v1",
            api_key_env="LITELLM_API_KEY",
            models=ModelConfig(
                reasoning="openai/gpt-4.1-mini",
                writing="openai/gpt-4.1-mini",
            ),
        )
    )
    session = backend.create_session(model="openai/gpt-4.1-mini")

    assert session.instruct("hello") == "LiteLLM answer"
    assert sink[0]["headers"]["Authorization"] == "Bearer secret-token"
