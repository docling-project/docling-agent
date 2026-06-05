"""Pytest configuration and shared fixtures for docling-agent tests.

This file is automatically discovered by pytest and makes fixtures
available to all test modules without explicit imports.
"""

import json
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument

from .test_utils import MockBackend


@pytest.fixture(scope="module")
def mock_backend() -> MockBackend:
    """Fixture providing a mocked backend instance."""
    return MockBackend()


@pytest.fixture(scope="module")
def test_document() -> DoclingDocument:
    """Fixture providing the test document loaded from JSON."""
    json_path = Path("tests/data/2408.09869v5.json")
    with open(json_path) as f:
        doc_dict = json.load(f)
    return DoclingDocument.model_validate(doc_dict)
