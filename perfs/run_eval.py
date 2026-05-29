from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval import EvalResult, evaluate


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
    print("field\tprecision\trecall\tf1\ttp\tfp\tfn")
    for field_result in result.field_metrics:
        print(
            f"{field_result.field}\t"
            f"{field_result.precision:.4f}\t"
            f"{field_result.recall:.4f}\t"
            f"{field_result.f1:.4f}\t"
            f"{field_result.tp}\t{field_result.fp}\t{field_result.fn}"
        )

    print("")
    print(f"macro\t{result.macro_precision:.4f}\t{result.macro_recall:.4f}\t{result.macro_f1:.4f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate extraction quality against ground-truth JSON files.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing predicted extraction JSON files.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Directory containing ground-truth JSON files.",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy value matching.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for fuzzy matching (0.0 to 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for EvalResult JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not 0.0 <= args.fuzzy_threshold <= 1.0:
        parser.error("--fuzzy-threshold must be between 0.0 and 1.0")

    try:
        pred_map = _load_json_map(args.predictions)
        gt_map = _load_json_map(args.ground_truth)
    except ValueError as exc:
        parser.error(str(exc))

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
        parser.error(" ; ".join(details))

    ordered_names = sorted(pred_names)
    predictions_list = [pred_map[name] for name in ordered_names]
    ground_truth_list = [gt_map[name] for name in ordered_names]

    result = evaluate(
        predictions_list,
        ground_truth_list,
        fuzzy=args.fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    _print_report(result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved evaluation JSON to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
