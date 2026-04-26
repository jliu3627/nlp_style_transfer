"""CLI for evaluating rewrite systems on detoxification metrics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from toxic_style_transfer.evaluation import RewriteEvaluator, RewriteSystemSpec, load_parallel_dataset


def main() -> None:
    args = _parse_args()
    dataset = load_parallel_dataset(args.input_csv, limit=args.limit)
    systems = _parse_systems(args.system)
    evaluator = RewriteEvaluator()
    report = evaluator.evaluate_systems(dataset=dataset, systems=systems)

    output_path = args.output or _default_output_path(args.output_dir)
    written = evaluator.write_report(report, output_path)
    print(f"Wrote detailed report to {written['json']}")
    print(f"Wrote summary table to {written['summary_csv']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detoxification rewrite systems.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with source,target columns.")
    parser.add_argument(
        "--system",
        action="append",
        default=[],
        help="Rewrite system spec in the form name=model_or_path. Repeat for multiple systems.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluations"),
        help="Directory for generated evaluation files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit JSON report path.",
    )
    return parser.parse_args()


def _parse_systems(system_args: List[str]) -> List[RewriteSystemSpec]:
    if not system_args:
        system_args = ["baseline=Ribin/t5-base_detoxParaphraser"]

    systems: List[RewriteSystemSpec] = []
    for item in system_args:
        if "=" not in item:
            raise ValueError(f"System spec must be name=model_or_path, got: {item}")
        name, value = item.split("=", 1)
        systems.append(RewriteSystemSpec(name=name.strip(), model_name_or_path=value.strip()))
    return systems


def _default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"rewrite_evaluation_{timestamp}.json"


if __name__ == "__main__":
    main()
