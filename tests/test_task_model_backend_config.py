from pathlib import Path

import pytest
from pydantic import ValidationError

from docling_agent.task_model import (
    AddTask,
    AgentTask,
    BackendConfig,
    ClearTask,
    ListTask,
    ViewTask,
    load_task,
)


def test_load_task_with_top_level_backend_block(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
query: "Summarize this document"
backend:
  type: lmstudio
  base_url: http://localhost:1234/v1
  models:
    reasoning: granite-3.3-8b-instruct
    writing: granite-3.3-8b-instruct
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, AgentTask)
    assert task.backend.type == "lmstudio"
    assert task.backend.base_url == "http://localhost:1234/v1"
    assert task.backend.models.reasoning == "granite-3.3-8b-instruct"


def test_load_add_task_without_query(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
mode: add
sources:
  - ./document.pdf
project_id: alpha
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, AddTask)
    assert task.query == ""
    assert task.project_id == "alpha"
    assert task.conversion == "standard"


def test_load_add_task_with_conversion_preset(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
mode: add
sources:
  - ./document.pdf
conversion: expensive
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, AddTask)
    assert task.conversion == "expensive"


def test_load_list_task_with_postgres_filter(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
mode: list
postgres_filter: "project_id = 'alpha'"
limit: 5
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, ListTask)
    assert task.postgres_filter == "project_id = 'alpha'"
    assert task.limit == 5


def test_view_task_requires_postgres_filter(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
mode: view
postgres_filter: " "
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="postgres_filter"):
        load_task(task_path)

    task_path.write_text(
        """
mode: view
postgres_filter: "project_id = 'alpha'"
""".strip(),
        encoding="utf-8",
    )
    assert isinstance(load_task(task_path), ViewTask)


def test_load_clear_task(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
mode: clear
project_id: alpha
all_projects: false
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert isinstance(task, ClearTask)
    assert task.project_id == "alpha"
    assert task.all_projects is False


def test_load_task_normalizes_yaml_query_param_scalars(tmp_path: Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
query: "Summarize this document"
backend:
  type: litellm
  query_params:
    api-version: 2024-02-01
    generated-at: 2024-02-01T12:30:00Z
    enabled: true
    disabled: false
    retries: 3
    temperature: 0.25
    optional:
""".strip(),
        encoding="utf-8",
    )

    task = load_task(task_path)

    assert task.backend.query_params == {
        "api-version": "2024-02-01",
        "generated-at": "2024-02-01T12:30:00+00:00",
        "enabled": "true",
        "disabled": "false",
        "retries": "3",
        "temperature": "0.25",
        "optional": "",
    }


@pytest.mark.parametrize(
    "value",
    [["value"], {"nested": "value"}, object()],
    ids=["list", "mapping", "object"],
)
def test_backend_config_rejects_non_scalar_query_param_values(value: object):
    with pytest.raises(ValidationError):
        BackendConfig.model_validate({"query_params": {"key": value}})
