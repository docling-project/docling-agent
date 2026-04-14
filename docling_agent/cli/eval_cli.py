from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from docling_agent.eval import EvalResult, evaluate

eval_app = typer.Typer(help="Evaluate extraction quality against ground-truth JSON files.")


def _load_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _load_json_map(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {directory}")
    files = sorted(directory.glob("*.json"))
    return {path.name: _load_json_file(path) for path in files}


def _print_report(result: EvalResult) -> None:
    typer.echo("field\tprecision\trecall\tf1\ttp\tfp\tfn")
    for field_result in result.field_metrics:
        typer.echo(
            f"{field_result.field}\t"
            f"{field_result.precision:.4f}\t"
            f"{field_result.recall:.4f}\t"
            f"{field_result.f1:.4f}\t"
            f"{field_result.tp}\t{field_result.fp}\t{field_result.fn}"
        )

    typer.echo("")
    typer.echo(
        "macro\t"
        f"{result.macro_precision:.4f}\t"
        f"{result.macro_recall:.4f}\t"
        f"{result.macro_f1:.4f}"
    )


@eval_app.callback(invoke_without_command=True)
def eval_main(
    predictions: Path = typer.Option(..., "--predictions", help="Directory containing predicted extraction JSON files."),
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Directory containing ground-truth JSON files."),
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Enable fuzzy value matching."),
    fuzzy_threshold: float = typer.Option(
        0.85,
        "--fuzzy-threshold",
        min=0.0,
        max=1.0,
        help="Similarity threshold for fuzzy matching.",
    ),
    output: Path | None = typer.Option(None, "--output", help="Optional output path for EvalResult JSON."),
) -> None:
    pred_map = _load_json_map(predictions)
    gt_map = _load_json_map(ground_truth)

    pred_names = set(pred_map)
    gt_names = set(gt_map)
    missing_in_pred = sorted(gt_names - pred_names)
    missing_in_gt = sorted(pred_names - gt_names)
    if missing_in_pred or missing_in_gt:
        details = []
        if missing_in_pred:
            details.append(f"Missing predictions for: {', '.join(missing_in_pred)}")
        if missing_in_gt:
            details.append(f"Missing ground truth for: {', '.join(missing_in_gt)}")
        raise typer.BadParameter(" ; ".join(details))

    ordered_names = sorted(pred_names)
    predictions_list = [pred_map[name] for name in ordered_names]
    ground_truth_list = [gt_map[name] for name in ordered_names]

    result = evaluate(
        predictions_list,
        ground_truth_list,
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
    )
    _print_report(result)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"Saved evaluation JSON to {output}")
