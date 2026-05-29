from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field


class FieldMetrics(BaseModel):
    field: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class EvalResult(BaseModel):
    field_metrics: list[FieldMetrics] = Field(default_factory=list)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0


def normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def values_match(pred: Any, gt: Any, *, fuzzy: bool, threshold: float) -> bool:
    normalized_pred = normalize(pred)
    normalized_gt = normalize(gt)

    if normalized_pred == normalized_gt:
        return True
    if not fuzzy:
        return False
    return SequenceMatcher(None, normalized_pred, normalized_gt).ratio() >= threshold


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.85,
) -> EvalResult:
    if len(predictions) != len(ground_truths):
        msg = (
            "Predictions and ground truths must have the same number of items. "
            f"Got predictions={len(predictions)}, ground_truths={len(ground_truths)}."
        )
        raise ValueError(msg)

    all_fields: set[str] = set()
    for item in predictions:
        all_fields.update(item.keys())
    for item in ground_truths:
        all_fields.update(item.keys())

    metric_counts: dict[str, dict[str, int]] = {field: {"tp": 0, "fp": 0, "fn": 0} for field in sorted(all_fields)}

    for pred_item, gt_item in zip(predictions, ground_truths):
        for field in metric_counts:
            pred_value = pred_item.get(field)
            gt_value = gt_item.get(field)
            pred_present = pred_value is not None
            gt_present = gt_value is not None

            if not pred_present and not gt_present:
                continue

            if pred_present and gt_present:
                if values_match(pred_value, gt_value, fuzzy=fuzzy, threshold=fuzzy_threshold):
                    metric_counts[field]["tp"] += 1
                else:
                    metric_counts[field]["fp"] += 1
                    metric_counts[field]["fn"] += 1
                continue

            if pred_present:
                metric_counts[field]["fp"] += 1
            else:
                metric_counts[field]["fn"] += 1

    field_metrics: list[FieldMetrics] = []
    for field, counts in metric_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _compute_f1(precision, recall)
        field_metrics.append(
            FieldMetrics(
                field=field,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    macro_precision = _safe_div(sum(item.precision for item in field_metrics), len(field_metrics))
    macro_recall = _safe_div(sum(item.recall for item in field_metrics), len(field_metrics))
    macro_f1 = _safe_div(sum(item.f1 for item in field_metrics), len(field_metrics))

    return EvalResult(
        field_metrics=field_metrics,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )
