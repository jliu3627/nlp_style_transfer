"""Command-line entry point for toxicity classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toxic_style_transfer.classifier import ToxicityClassifier


def main() -> None:
    args = _parse_args()
    classifier = ToxicityClassifier(provider=args.provider)

    texts = _load_texts(args)
    results = [classifier.classify(text).to_dict() for text in texts]

    if args.jsonl:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    else:
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify text as toxic or non-toxic.")
    parser.add_argument("text", nargs="?", help="Text to classify.")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional file containing one text example per line.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "heuristic"),
        default="openai",
        help="Classification backend to use.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one JSON object per line for batch input.",
    )
    return parser.parse_args()


def _load_texts(args: argparse.Namespace) -> list[str]:
    if args.input_file:
        lines = args.input_file.read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in lines if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        raise SystemExit("Provide text or --input-file.")

    if not texts:
        raise SystemExit("No non-empty texts to classify.")
    return texts


if __name__ == "__main__":
    main()
