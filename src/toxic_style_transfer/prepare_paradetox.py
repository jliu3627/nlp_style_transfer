"""Download and split the ParaDetox dataset into local CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("s-nlp/paradetox", split="train")
    dataset = dataset.rename_columns(
        {
            "en_toxic_comment": "source",
            "en_neutral_comment": "target",
        }
    )
    dataset = dataset.select_columns(["source", "target"])

    first_split = dataset.train_test_split(test_size=args.eval_size + args.test_size, seed=args.seed)
    holdout_fraction = args.test_size / (args.eval_size + args.test_size)
    second_split = first_split["test"].train_test_split(test_size=holdout_fraction, seed=args.seed)

    train_dataset = first_split["train"]
    eval_dataset = second_split["train"]
    test_dataset = second_split["test"]

    train_path = output_dir / "paradetox_train.csv"
    eval_path = output_dir / "paradetox_eval.csv"
    test_path = output_dir / "paradetox_test.csv"

    train_dataset.to_csv(str(train_path), index=False)
    eval_dataset.to_csv(str(eval_path), index=False)
    test_dataset.to_csv(str(test_path), index=False)

    print(f"Wrote {len(train_dataset)} rows to {train_path}")
    print(f"Wrote {len(eval_dataset)} rows to {eval_path}")
    print(f"Wrote {len(test_dataset)} rows to {test_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and split the ParaDetox dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/paradetox"))
    parser.add_argument("--eval-size", type=float, default=0.1, help="Fraction of full data for eval split.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of full data for test split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
