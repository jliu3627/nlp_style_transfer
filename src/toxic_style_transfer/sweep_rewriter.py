"""Run a hyperparameter sweep for the Seq2Seq rewriter."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import pandas as pd

from toxic_style_transfer.evaluation import RewriteEvaluator, RewriteSystemSpec, load_parallel_dataset
from toxic_style_transfer.train_rewriter import train


DEFAULT_SWEEP_MODEL_NAMES: Sequence[str] = ("google/flan-t5-base",)
DEFAULT_SWEEP_LEARNING_RATES: Sequence[float] = (5e-5, 3e-5, 2e-5)
DEFAULT_SWEEP_EPOCHS: Sequence[int] = (3, 5, 8)
DEFAULT_SWEEP_BATCH_SIZES: Sequence[int] = (4, 8)
DEFAULT_SWEEP_SEEDS: Sequence[int] = (42, 123)
DEFAULT_BASELINE_MODEL = "Ribin/t5-base_detoxParaphraser"


@dataclass(frozen=True)
class SweepTrial:
    model_name: str
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int
    validation_split: float
    max_input_length: int
    max_target_length: int

    @property
    def name(self) -> str:
        model_slug = _slugify_model_name(self.model_name)
        lr_slug = str(self.learning_rate).replace("-", "m").replace(".", "p")
        return (
            f"{model_slug}_lr{lr_slug}_e{self.epochs}_b{self.batch_size}"
            f"_seed{self.seed}"
        )


def main() -> None:
    args = _parse_args()
    trials = _build_trials(args)
    summary_rows = run_sweep(args, trials)
    _write_sweep_summary(args.summary_path, summary_rows)
    print(f"Wrote sweep summary to {args.summary_path}")


def run_sweep(args: argparse.Namespace, trials: Sequence[SweepTrial]) -> list[dict[str, Any]]:
    evaluator = None
    eval_dataset = None
    if not args.skip_eval:
        eval_dataset = load_parallel_dataset(args.eval_input_csv, limit=args.eval_limit)

    summary_rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        print(f"[{index}/{len(trials)}] Running {trial.name}")
        output_dir = args.output_root / trial.name

        row = asdict(trial)
        row["trial_name"] = trial.name
        row["output_dir"] = str(output_dir)
        try:
            if output_dir.exists() and (output_dir / "model.safetensors").exists() and not args.overwrite:
                print(f"  Skipping training because {output_dir} already exists.")
            else:
                train(_training_namespace(args, trial, output_dir))

            row["status"] = "trained"
            if not args.skip_eval and eval_dataset is not None:
                if evaluator is None:
                    evaluator = RewriteEvaluator()
                report = evaluator.evaluate_systems(
                    dataset=eval_dataset,
                    systems=[
                        RewriteSystemSpec(name="baseline", model_name_or_path=args.baseline_model),
                        RewriteSystemSpec(name=trial.name, model_name_or_path=str(output_dir)),
                    ],
                )
                output_path = _trial_report_path(args.report_dir, trial.name)
                written = evaluator.write_report(report, output_path)
                row["report_json"] = str(written["json"])
                row["report_summary_csv"] = str(written["summary_csv"])
                baseline_metrics = report["systems"][0]["aggregate_metrics"]
                trial_metrics = report["systems"][1]["aggregate_metrics"]
                row.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
                row.update(trial_metrics)
                row["status"] = "evaluated"
        except Exception as exc:  # pragma: no cover - we want the sweep to keep going
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(f"  Trial failed: {exc}")

        summary_rows.append(row)
        _write_sweep_summary(args.summary_path, summary_rows)

    return summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for the rewriter model.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with source,target columns.")
    parser.add_argument("--output-root", type=Path, default=Path("models/sweeps"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/evaluations/sweeps"))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/evaluations/sweeps/sweep_summary.csv"),
        help="CSV path for incremental sweep results.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=None,
        help="Repeat to override the default sweep model list.",
    )
    parser.add_argument("--learning-rate", action="append", type=float, default=None)
    parser.add_argument("--epochs", action="append", type=int, default=None)
    parser.add_argument("--batch-size", action="append", type=int, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--skip-eval", action="store_true", help="Train all trials without running evaluation.")
    parser.add_argument(
        "--eval-input-csv",
        type=Path,
        default=Path("data/paradetox/paradetox_test.csv"),
        help="Held-out CSV used for sweep evaluation.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=200,
        help="Optional row cap for faster sweep scoring. Use 0 or a negative number for the full dataset.",
    )
    parser.add_argument(
        "--baseline-model",
        default=DEFAULT_BASELINE_MODEL,
        help="Baseline rewrite model to compare against during sweep evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain trials even if a checkpoint already exists in the output directory.",
    )
    return parser.parse_args()


def _build_trials(args: argparse.Namespace) -> list[SweepTrial]:
    model_names = args.model_name or DEFAULT_SWEEP_MODEL_NAMES
    learning_rates = args.learning_rate or DEFAULT_SWEEP_LEARNING_RATES
    epochs = args.epochs or DEFAULT_SWEEP_EPOCHS
    batch_sizes = args.batch_size or DEFAULT_SWEEP_BATCH_SIZES
    seeds = args.seed or DEFAULT_SWEEP_SEEDS

    trials = [
        SweepTrial(
            model_name=model_name,
            learning_rate=learning_rate,
            epochs=epoch_count,
            batch_size=batch_size,
            seed=seed,
            validation_split=args.validation_split,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        )
        for model_name, learning_rate, epoch_count, batch_size, seed in product(
            model_names,
            learning_rates,
            epochs,
            batch_sizes,
            seeds,
        )
    ]

    if not trials:
        raise ValueError("Sweep produced no trials.")
    return trials


def _training_namespace(
    args: argparse.Namespace,
    trial: SweepTrial,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_csv=args.input_csv,
        output_dir=output_dir,
        model_name=trial.model_name,
        batch_size=trial.batch_size,
        epochs=trial.epochs,
        learning_rate=trial.learning_rate,
        max_input_length=trial.max_input_length,
        max_target_length=trial.max_target_length,
        validation_split=trial.validation_split,
        seed=trial.seed,
        activate_model=False,
    )


def _trial_report_path(report_dir: Path, trial_name: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{trial_name}.json"


def _write_sweep_summary(summary_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(summary_path, index=False)
    summary_path.with_suffix(".json").write_text(
        json.dumps(list(rows), indent=2, default=str),
        encoding="utf-8",
    )


def _slugify_model_name(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


if __name__ == "__main__":
    main()
