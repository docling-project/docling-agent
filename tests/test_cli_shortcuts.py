from __future__ import annotations

from pathlib import Path

import tomllib


def test_library_shortcut_scripts_are_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]

    assert scripts["add-sources"] == "docling_agent.cli:add_sources_cli"
    assert scripts["list-sources"] == "docling_agent.cli:list_sources_cli"
    assert scripts["view-sources"] == "docling_agent.cli:view_sources_cli"
    assert scripts["clear-sources"] == "docling_agent.cli:clear_sources_cli"
