"""Compare CNN and XGBoost test metrics and save a combined report.

Example:
    python -m src.pipeline.compare_models --cnn logs/metrics.json --xgb logs/xgb/metrics.json --out logs/compare
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

PREFERRED_METRICS = ["accuracy", "roc_auc", "pr_auc", "f1"]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_test_metrics(payload: Dict[str, object]) -> Dict[str, object]:
    metrics = payload.get("test")
    if metrics is None or not isinstance(metrics, dict):
        raise ValueError("Missing 'test' metrics in JSON payload.")
    return metrics


def resolve_metric_columns(metrics_list: Iterable[Dict[str, object]]) -> List[str]:
    common_keys = set.intersection(*(set(metrics.keys()) for metrics in metrics_list))
    ordered = [metric for metric in PREFERRED_METRICS if metric in common_keys]
    remaining = sorted(common_keys.difference(ordered))
    return ordered + remaining


def build_row(model_name: str, metrics: Dict[str, object], columns: List[str]) -> Dict[str, object]:
    row = {"model": model_name}
    for column in columns:
        row[column] = metrics.get(column)
    return row


def write_outputs(rows: List[Dict[str, object]], columns: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *columns])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN and XGBoost test metrics")
    parser.add_argument("--cnn", type=Path, required=True, help="Path to CNN metrics.json file")
    parser.add_argument("--xgb", type=Path, required=True, help="Path to XGBoost metrics.json file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/compare"),
        help="Output directory for comparison metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cnn_metrics = extract_test_metrics(load_json(args.cnn))
    xgb_metrics = extract_test_metrics(load_json(args.xgb))

    columns = resolve_metric_columns([cnn_metrics, xgb_metrics])
    rows = [
        build_row("cnn", cnn_metrics, columns),
        build_row("xgb", xgb_metrics, columns),
    ]

    write_outputs(rows, columns, args.out)


if __name__ == "__main__":
    main()
