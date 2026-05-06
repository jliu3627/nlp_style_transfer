"""Train a T5 rewriter on toxic-to-neutral parallel data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from dotenv import dotenv_values
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


DEFAULT_MODEL_NAME = "google/flan-t5-large"
DEFAULT_SAVE_TOTAL_LIMIT = 2


def main() -> None:
    args = _parse_args()
    train(args)


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_df, eval_df = _load_train_and_eval_dataframes(args)
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    def preprocess(example):
        model_input = "rewrite toxic to neutral: " + example["source"]
        inputs = tokenizer(
            model_input,
            max_length=args.max_input_length,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            text_target=example["target"],
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics["train_rows"] = len(train_df)
    metrics["eval_rows"] = len(eval_df)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    _write_training_summary(args.output_dir, metrics)
    if args.activate_model:
        _set_rewrite_model_path(args.output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Seq2Seq toxic-to-neutral rewriter.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with source,target columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/rewriter"), help="Checkpoint output directory.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activate-model",
        action="store_true",
        help="After successful training, write REWRITE_MODEL_PATH to the local .env file.",
    )
    return parser.parse_args()


def _load_parallel_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df)
    return df[["source", "target"]].copy()


def _load_train_and_eval_dataframes(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _load_parallel_dataframe(args.input_csv)

    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split must be between 0 and 1.")

    if len(train_df) < 2:
        raise ValueError("Need at least 2 rows to create a validation split.")

    eval_rows = max(1, int(round(len(train_df) * args.validation_split)))
    eval_rows = min(eval_rows, len(train_df) - 1)
    shuffled = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_df = shuffled.iloc[:eval_rows].reset_index(drop=True)
    remaining_train_df = shuffled.iloc[eval_rows:].reset_index(drop=True)
    return remaining_train_df, eval_df


def _write_training_summary(output_dir: Path, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def _set_rewrite_model_path(output_dir: Path, env_path: Path = Path(".env")) -> None:
    resolved_output_dir = output_dir.as_posix()
    existing_values = dotenv_values(env_path) if env_path.exists() else {}
    existing_values["REWRITE_MODEL_PATH"] = resolved_output_dir

    lines = [f"{key}={value}" for key, value in existing_values.items() if value is not None]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"source", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


if __name__ == "__main__":
    main()
