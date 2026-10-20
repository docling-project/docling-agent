"""Tests for the import-time footprint of docling_agent."""

import subprocess
import sys


def test_import_does_not_load_document_extractor():
    """Importing the package must not require the VLM stack behind DocumentExtractor."""
    code = "import sys, docling_agent; sys.exit('docling.document_extractor' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or "docling.document_extractor was imported"
