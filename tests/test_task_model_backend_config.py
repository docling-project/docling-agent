from pathlib import Path

from docling_agent.task_model import AgentTask, load_task


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
