from __future__ import annotations

import json

from typer.testing import CliRunner

from docling_agent.cli import app
from docling_agent.eval import evaluate, values_match


def _metrics_by_field(result):
    return {item.field: item for item in result.field_metrics}


def test_evaluate_exact_match_scores_are_perfect():
    predictions = [{"name": "Alice", "amount": 10}]
    ground_truths = [{"name": "Alice", "amount": 10}]

    result = evaluate(predictions, ground_truths)
    metrics = _metrics_by_field(result)

    assert metrics["name"].f1 == 1.0
    assert metrics["amount"].f1 == 1.0
    assert result.macro_f1 == 1.0


def test_evaluate_complete_mismatch_scores_are_zero():
    predictions = [{"name": "Alice"}]
    ground_truths = [{"name": "Bob"}]

    result = evaluate(predictions, ground_truths)
    name_metrics = _metrics_by_field(result)["name"]

    assert name_metrics.tp == 0
    assert name_metrics.fp == 1
    assert name_metrics.fn == 1
    assert name_metrics.precision == 0.0
    assert name_metrics.recall == 0.0
    assert name_metrics.f1 == 0.0


def test_evaluate_partial_match_with_missing_and_extra_fields():
    predictions = [{"a": "x", "b": "y"}]
    ground_truths = [{"a": "x", "c": "z"}]

    result = evaluate(predictions, ground_truths)
    metrics = _metrics_by_field(result)

    assert metrics["a"].tp == 1
    assert metrics["b"].fp == 1
    assert metrics["c"].fn == 1


def test_values_match_normalizes_whitespace_and_case():
    assert values_match("  Hello   World ", "hello world", fuzzy=False, threshold=0.85)


def test_evaluate_fuzzy_mode_with_threshold():
    predictions = [{"vendor": "Acme Corporation"}]
    ground_truths = [{"vendor": "acme corp"}]

    exact_result = evaluate(predictions, ground_truths, fuzzy=False)
    fuzzy_result = evaluate(predictions, ground_truths, fuzzy=True, fuzzy_threshold=0.7)

    assert _metrics_by_field(exact_result)["vendor"].tp == 0
    assert _metrics_by_field(fuzzy_result)["vendor"].tp == 1


def test_evaluate_empty_inputs_returns_zero_macro_scores():
    result = evaluate([], [])
    assert result.field_metrics == []
    assert result.macro_precision == 0.0
    assert result.macro_recall == 0.0
    assert result.macro_f1 == 0.0


def test_evaluate_multi_document_aggregation():
    predictions = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    ground_truths = [{"name": "A"}, {"name": "X"}, {}]

    result = evaluate(predictions, ground_truths)
    name_metrics = _metrics_by_field(result)["name"]

    assert name_metrics.tp == 1
    assert name_metrics.fp == 2
    assert name_metrics.fn == 1


def test_eval_cli_runs_and_writes_output(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    (pred_dir / "doc1.json").write_text(json.dumps({"field": "Value"}), encoding="utf-8")
    (gt_dir / "doc1.json").write_text(json.dumps({"field": "value"}), encoding="utf-8")

    output = tmp_path / "result.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--predictions",
            str(pred_dir),
            "--ground-truth",
            str(gt_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "macro" in result.output
    assert output.exists()
