from docling_agent.agents import EvalResult, evaluate

PREDICTIONS = [
    {"invoice-number": "INV-001", "vendor": "Acme Corporation", "total": 100.0, "currency": "USD"},
    {"invoice-number": "inv-002", "vendor": "Globex", "total": 250.0, "currency": "EUR"},
    {"invoice-number": "INV-003", "vendor": "Initech", "total": 75.5},
]

GROUND_TRUTH = [
    {"invoice-number": "INV-001", "vendor": "acme corp", "total": 100.0, "currency": "USD"},
    {"invoice-number": "INV-002", "vendor": "Globex", "total": 250.0, "currency": "EUR"},
    {"invoice-number": "INV-003", "vendor": "Initech Inc.", "total": 80.0, "currency": "USD"},
]


def print_report(title: str, result: EvalResult) -> None:
    print(f"\n=== {title} ===")
    print(f"{'field':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'tp':>5}{'fp':>5}{'fn':>5}")
    for m in result.field_metrics:
        print(
            f"{m.field:<16}{m.precision:>10.2f}{m.recall:>10.2f}{m.f1:>10.2f}"
            f"{m.tp:>5}{m.fp:>5}{m.fn:>5}"
        )
    print(
        f"{'macro':<16}{result.macro_precision:>10.2f}"
        f"{result.macro_recall:>10.2f}{result.macro_f1:>10.2f}"
    )


def main() -> None:
    print_report("exact", evaluate(PREDICTIONS, GROUND_TRUTH))
    print_report(
        "fuzzy (0.7)",
        evaluate(PREDICTIONS, GROUND_TRUTH, fuzzy=True, fuzzy_threshold=0.7),
    )


if __name__ == "__main__":
    main()
