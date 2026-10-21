"""Conversion of source files into DoclingDocuments, locally or with a Docling service.

The base install of docling-agent depends on ``docling-slim`` only, which can convert sources through a
Docling Serve endpoint (for example Docling for IBM watsonx). Converting them in-process needs the
local models and parsers, which come with the ``local`` extra (``pip install 'docling-agent[local]'``).
"""

from __future__ import annotations

import importlib.util
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Literal

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from docling_agent.logging import log_info

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling.service_client import DoclingServiceClient

ConversionPreset = Literal["fast", "standard", "expensive"]

LOCAL_EXTRA_HINT = "Install the local extra with: pip install 'docling-agent[local]'"


class ConversionMode(str, Enum):
    """Where source files are converted."""

    AUTO = "auto"
    """Use the Docling service when ``DOCLING_AGENT_SERVICE_URL`` is set, convert locally otherwise."""

    LOCAL = "local"
    """Convert in-process with docling (requires the ``local`` extra)."""

    REMOTE = "remote"
    """Convert with a Docling Serve endpoint."""


class ConversionSettings(BaseSettings):
    """Settings for converting source files, read from ``DOCLING_AGENT_*`` environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_prefix="DOCLING_AGENT_", env_file=".env", extra="ignore")

    conversion_mode: Annotated[
        ConversionMode,
        Field(description="Where source files are converted: auto, local or remote."),
    ] = ConversionMode.AUTO
    service_url: Annotated[
        str | None,
        Field(description="Base URL of the Docling Serve endpoint used for remote conversion."),
    ] = None
    service_api_key: Annotated[
        str | None,
        Field(repr=False, description="API key for the Docling Serve endpoint."),
    ] = None
    service_timeout: Annotated[
        float,
        Field(description="Timeout in seconds for a remote conversion."),
    ] = 300.0

    def resolved_mode(self) -> ConversionMode:
        """Return the conversion mode, resolving ``auto`` from the presence of a service URL."""
        if self.conversion_mode == ConversionMode.AUTO:
            return ConversionMode.REMOTE if self.service_url else ConversionMode.LOCAL
        return self.conversion_mode


def local_conversion_available() -> bool:
    """Return whether the local conversion stack (models and parsers) is installed."""
    return importlib.util.find_spec("docling_ibm_models") is not None


class LocalSourceConverter:
    """Convert source files in-process with docling's DocumentConverter."""

    def __init__(self, preset: ConversionPreset) -> None:
        """Build a DocumentConverter configured for the given preset.

        Args:
            preset: The named conversion preset.

        Raises:
            ImportError: If the ``local`` extra is not installed.
        """
        if not local_conversion_available():
            raise ImportError(
                "Converting source files locally needs the local docling models and parsers. "
                f"{LOCAL_EXTRA_HINT}, or set DOCLING_AGENT_SERVICE_URL to convert with a Docling service."
            )
        from docling.document_converter import DocumentConverter

        self.preset = preset
        self.converter: DocumentConverter = DocumentConverter()
        for format_option in self.converter.format_to_options.values():
            pipeline_options = format_option.pipeline_options
            if pipeline_options is None:
                continue
            if hasattr(pipeline_options, "do_ocr"):
                pipeline_options.do_ocr = preset != "fast"
            if hasattr(pipeline_options, "do_table_structure"):
                pipeline_options.do_table_structure = preset != "fast"
            if hasattr(pipeline_options, "do_picture_classification"):
                pipeline_options.do_picture_classification = True
            if hasattr(pipeline_options, "do_chart_extraction"):
                pipeline_options.do_chart_extraction = preset == "expensive"
            if hasattr(pipeline_options, "generate_page_images"):
                pipeline_options.generate_page_images = True

    def convert(self, source: Path) -> ConversionResult:
        """Convert one source file."""
        return self.converter.convert(source)

    def pipeline_name(self, input_format: InputFormat) -> str:
        """Return the name of the pipeline used for an input format, recorded in the library."""
        format_option = self.converter.format_to_options.get(input_format)
        if format_option is None:
            return f"unknown:{self.preset}"
        return f"{format_option.pipeline_cls.__name__}:{self.preset}"

    def close(self) -> None:
        """Release resources (nothing to release for local conversion)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class RemoteSourceConverter:
    """Convert source files with a Docling Serve endpoint."""

    def __init__(self, preset: ConversionPreset, settings: ConversionSettings) -> None:
        """Create a service client and the request options for the given preset.

        Args:
            preset: The named conversion preset.
            settings: The conversion settings holding the service URL and API key.

        Raises:
            ValueError: If no service URL is configured.
        """
        if not settings.service_url:
            raise ValueError(
                "DOCLING_AGENT_CONVERSION_MODE is 'remote' but DOCLING_AGENT_SERVICE_URL is not set. "
                f"Set the service URL, or convert locally ({LOCAL_EXTRA_HINT})."
            )
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.service_client import DoclingServiceClient

        self.preset = preset
        self.options = ConvertDocumentsOptions(
            do_ocr=preset != "fast",
            do_table_structure=preset != "fast",
            do_picture_classification=True,
            do_chart_extraction=preset == "expensive",
            include_images=True,
        )
        self.client: DoclingServiceClient = DoclingServiceClient(
            url=settings.service_url,
            api_key=settings.service_api_key or "",
            job_timeout=settings.service_timeout,
        )

    def convert(self, source: Path) -> ConversionResult:
        """Convert one source file."""
        return self.client.convert(source=source, options=self.options)

    def pipeline_name(self, input_format: InputFormat) -> str:
        """Return the name of the pipeline used for an input format, recorded in the library."""
        return f"DoclingServiceClient:{self.preset}"

    def close(self) -> None:
        """Close the service client."""
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


SourceConverter = LocalSourceConverter | RemoteSourceConverter


def create_source_converter(preset: ConversionPreset, settings: ConversionSettings | None = None) -> SourceConverter:
    """Return a local or remote converter for the given preset, according to the settings.

    Args:
        preset: The named conversion preset.
        settings: The conversion settings. Read from the environment when omitted.

    Returns:
        A converter usable as a context manager.
    """
    settings = settings or ConversionSettings()
    mode = settings.resolved_mode()
    log_info(f"Converting sources with the {mode.value} converter (preset: {preset})")
    if mode == ConversionMode.REMOTE:
        return RemoteSourceConverter(preset, settings)
    return LocalSourceConverter(preset)
