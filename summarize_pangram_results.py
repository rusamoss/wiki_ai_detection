#!/usr/bin/env python3
"""
summarize_pangram_results.py

Summarizes a run_pangram.py results CSV (default: data/pangram_results.csv):
for each label, the number of articles and what percent Pangram classified
as each prediction_short value (e.g. Human/Mixed/AI).

Usage:
    python summarize_pangram_results.py
"""

import argparse
import csv
from collections import Counter

from get_test_wiki_pages import DATA_DIR

DEFAULT_RESULTS_CSV = f"{DATA_DIR}/pangram_results.csv"


def distinct_predictions(rows: list[dict[str, str]]) -> list[str]:
    """Returns the distinct prediction_short values found in rows, sorted --
    whatever categories Pangram actually returned, not an assumed list."""
    return sorted({row["prediction_short"] for row in rows if row["prediction_short"]})


def summarize(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    """Returns {label: Counter({prediction_short: n, ..., "error": n})},
    grouping rows with no prediction_short (a failed Pangram call) under "error"."""
    by_label: dict[str, Counter[str]] = {}
    for row in rows:
        counts = by_label.setdefault(row["label"], Counter())
        counts[row["prediction_short"] or "error"] += 1
    return by_label


def print_summary(by_label: dict[str, Counter[str]], predictions: list[str]) -> None:
    width = max(len(p) for p in [*predictions, "error"])
    for label in sorted(by_label):
        counts = by_label[label]
        total = sum(counts.values())
        print(f"{label} ({total} article{'s' if total != 1 else ''})")
        for prediction in predictions:
            n = counts.get(prediction, 0)
            pct = 100 * n / total if total else 0.0
            print(f"  {prediction:<{width}} {n:>4} ({pct:5.1f}%)")
        errors = counts.get("error", 0)
        if errors:
            print(f"  {'error':<{width}} {errors:>4} ({100 * errors / total:5.1f}%)")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        help=f"Pangram results CSV to summarize (default: {DEFAULT_RESULTS_CSV})",
    )
    args = ap.parse_args()

    with open(args.results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows found in {args.results_csv}")

    print_summary(summarize(rows), distinct_predictions(rows))


if __name__ == "__main__":
    main()
