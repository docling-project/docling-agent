from __future__ import annotations

from pathlib import Path
from subprocess import run

import tomllib


def test_library_shortcut_scripts_are_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]

    assert scripts["add-sources"] == "docling_agent.cli:add_sources_cli"
    assert scripts["list-sources"] == "docling_agent.cli:list_sources_cli"
    assert scripts["view-sources"] == "docling_agent.cli:view_sources_cli"
    assert scripts["clear-sources"] == "docling_agent.cli:clear_sources_cli"
    assert scripts["compile-sources"] == "docling_agent.cli:compile_sources_cli"


def test_shortcut_command_supports_short_help_option() -> None:
    result = run(["uv", "run", "compile-sources", "-h"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "--help" in result.stdout
    assert "-h" in result.stdout
    assert "--task" in result.stdout
    assert "-t" in result.stdout
    assert "--llm-review-terms" not in result.stdout
