"""Train a T5 rewriter on toxic-to-neutral parallel data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


DEFAULT_MODEL_NAME = "google/flan-t5-small"


def main() -> None:
    args = _parse_args()
    train(args)


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(args.input_csv)
    _validate_columns(df)
    dataset = Dataset.from_pandas(df[["source", "target"]])

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

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


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
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"source", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


if __name__ == "__main__":
    main()
