"""Tests for converting source files locally or with a Docling service."""

import pytest

from docling_agent import conversion
from docling_agent.conversion import (
    ConversionMode,
    ConversionSettings,
    LocalSourceConverter,
    RemoteSourceConverter,
    create_source_converter,
)


def _settings(**kwargs) -> ConversionSettings:
    # Ignore the environment and any .env file, so tests only see what they pass
    return ConversionSettings(_env_file=None, **kwargs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CONVERSION_MODE", "SERVICE_URL", "SERVICE_API_KEY", "SERVICE_TIMEOUT"):
        monkeypatch.delenv(f"DOCLING_AGENT_{name}", raising=False)


def test_auto_mode_uses_remote_when_service_url_is_set() -> None:
    assert _settings(service_url="https://docling.example.com").resolved_mode() == ConversionMode.REMOTE


def test_auto_mode_uses_local_without_service_url() -> None:
    assert _settings().resolved_mode() == ConversionMode.LOCAL


def test_explicit_mode_wins_over_service_url() -> None:
    settings = _settings(conversion_mode="local", service_url="https://docling.example.com")
    assert settings.resolved_mode() == ConversionMode.LOCAL


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_AGENT_SERVICE_URL", "https://docling.example.com")
    monkeypatch.setenv("DOCLING_AGENT_SERVICE_API_KEY", "secret")
    settings = _settings()
    assert settings.service_url == "https://docling.example.com"
    assert settings.service_api_key == "secret"
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    ("preset", "heavy", "charts"),
    [("fast", False, False), ("standard", True, False), ("expensive", True, True)],
)
def test_remote_converter_maps_presets_to_options(preset, heavy, charts) -> None:
    converter = create_source_converter(preset, _settings(service_url="https://docling.example.com"))
    with converter:
        assert isinstance(converter, RemoteSourceConverter)
        assert converter.options.do_ocr is heavy
        assert converter.options.do_table_structure is heavy
        assert converter.options.do_picture_classification is True
        assert converter.options.do_chart_extraction is charts
        assert converter.pipeline_name(input_format=None) == f"DoclingServiceClient:{preset}"


def test_remote_converter_requires_service_url() -> None:
    with pytest.raises(ValueError, match="DOCLING_AGENT_SERVICE_URL"):
        create_source_converter("standard", _settings(conversion_mode="remote"))


def test_local_converter_requires_local_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversion, "local_conversion_available", lambda: False)
    with pytest.raises(ImportError, match=r"docling-agent\[local\]"):
        create_source_converter("standard", _settings())


@pytest.mark.skipif(not conversion.local_conversion_available(), reason="requires the local extra")
def test_local_converter_is_used_without_service_url() -> None:
    with create_source_converter("fast", _settings()) as converter:
        assert isinstance(converter, LocalSourceConverter)
