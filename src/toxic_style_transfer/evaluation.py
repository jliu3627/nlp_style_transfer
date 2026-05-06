"""Evaluation utilities for detoxification rewrite models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

from toxic_style_transfer.rewriter import Seq2SeqRewriter


DEFAULT_FORMALITY_MODEL = "s-nlp/roberta-base-formality-ranker"
DEFAULT_BARTSCORE_CHECKPOINT = "facebook/bart-base"
DEFAULT_BERTSCORE_MODEL = "roberta-base"


def _default_device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_local_model_cache() -> None:
    cache_root = Path.cwd() / ".model_cache"
    torch_home = cache_root / "torch"
    hf_home = cache_root / "huggingface"
    torch_home.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))


def _evaluation_device() -> str:
    return _default_device_name()


@dataclass(frozen=True)
class RewriteSystemSpec:
    """A rewrite system to evaluate."""

    name: str
    model_name_or_path: str


class DetoxifyScorer:
    """Toxicity scorer based on the Detoxify model from the slides."""

    def __init__(self, model_name: str = "original") -> None:
        _ensure_local_model_cache()
        from detoxify import Detoxify

        self._model = Detoxify(model_name, device=_evaluation_device())

    def score(self, texts: Sequence[str]) -> List[float]:
        predictions = self._model.predict(list(texts))
        return [float(score) for score in predictions["toxicity"]]


class FormalityScorer:
    """Formality scorer based on the ranker cited in the slides."""

    def __init__(self, model_name: str = DEFAULT_FORMALITY_MODEL) -> None:
        _ensure_local_model_cache()
        self._device = torch.device(_evaluation_device())
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self._device)
        self._model.eval()
        label2id = {key.lower(): value for key, value in self._model.config.label2id.items()}
        self._formal_idx = label2id.get("formal", 1)

    def score(self, texts: Sequence[str], batch_size: int = 8) -> List[float]:
        scores: List[float] = []
        with torch.no_grad():
            for batch in _batched(texts, batch_size):
                encoded = self._tokenizer(
                    list(batch),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, self._formal_idx]
                scores.extend(float(value) for value in probs.cpu().tolist())
        return scores


class BERTScoreScorer:
    """Meaning-preservation scorer using BERTScore."""

    def __init__(self, model_type: str = DEFAULT_BERTSCORE_MODEL) -> None:
        _ensure_local_model_cache()
        from bert_score import BERTScorer

        device = _evaluation_device()
        self._scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            rescale_with_baseline=True,
            device=device,
        )

    def score(self, candidates: Sequence[str], references: Sequence[str]) -> Dict[str, List[float]]:
        precision, recall, f1 = self._scorer.score(list(candidates), list(references))
        return {
            "precision": [float(value) for value in precision.cpu().tolist()],
            "recall": [float(value) for value in recall.cpu().tolist()],
            "f1": [float(value) for value in f1.cpu().tolist()],
        }


class BARTFluencyScorer:
    """Fluency proxy based on BARTScore-style sequence likelihood."""

    def __init__(self, checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT) -> None:
        _ensure_local_model_cache()
        self._device = torch.device(_evaluation_device())
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self._device)
        self._model.eval()

    def score(self, texts: Sequence[str], batch_size: int = 4) -> List[float]:
        scores: List[float] = []
        with torch.no_grad():
            for batch in _batched(texts, batch_size):
                outputs = self._tokenizer(
                    list(batch),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
                labels = outputs["input_ids"].clone()
                labels[outputs["attention_mask"] == 0] = -100

                inputs = self._tokenizer(
                    [""] * len(batch),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=4,
                )
                model_inputs = {key: value.to(self._device) for key, value in inputs.items()}
                label_ids = labels.to(self._device)
                result = self._model(**model_inputs, labels=label_ids)

                token_losses = F.cross_entropy(
                    result.logits.view(-1, result.logits.size(-1)),
                    label_ids.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(label_ids.size(0), label_ids.size(1))
                token_mask = (label_ids != -100).float()
                batch_scores = -(token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
        return scores


class RewriteEvaluator:
    """Evaluate multiple rewrite systems on a toxic-to-neutral parallel dataset."""

    def __init__(
        self,
        toxicity_scorer: Optional[Any] = None,
        formality_scorer: Optional[Any] = None,
        bertscore_scorer: Optional[Any] = None,
        fluency_scorer: Optional[Any] = None,
    ) -> None:
        self._toxicity_scorer = toxicity_scorer or DetoxifyScorer()
        self._formality_scorer = formality_scorer or FormalityScorer()
        self._bertscore_scorer = bertscore_scorer or BERTScoreScorer()
        self._fluency_scorer = fluency_scorer or BARTFluencyScorer()

    def evaluate_systems(
        self,
        dataset: pd.DataFrame,
        systems: Sequence[RewriteSystemSpec],
    ) -> dict[str, Any]:
        records = dataset.to_dict(orient="records")
        results = []
        for spec in tqdm(list(systems), desc="Systems", unit="system"):
            results.append(self._evaluate_one_system(records, spec))
        return {
            "dataset_size": len(records),
            "systems": results,
        }

    def write_report(
        self,
        report: dict[str, Any],
        output_path: Path,
    ) -> dict[str, Path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        summary_rows = []
        for system in report["systems"]:
            row = {"system": system["system"]}
            row.update(system["aggregate_metrics"])
            summary_rows.append(row)

        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return {"json": output_path, "summary_csv": summary_path}

    def _evaluate_one_system(
        self,
        records: Sequence[dict[str, Any]],
        spec: RewriteSystemSpec,
    ) -> dict[str, Any]:
        rewriter = _build_rewriter(spec.model_name_or_path)
        sources = [str(record["source"]) for record in records]
        references = [str(record["target"]) for record in records]
        predictions = [
            rewriter.rewrite(source)
            for source in tqdm(
                sources,
                desc=f"Rewriting [{spec.name}]",
                unit="text",
            )
        ]

        tqdm.write(f"Scoring toxicity [{spec.name}]")
        toxicity_scores = self._toxicity_scorer.score(predictions)
        tqdm.write(f"Scoring formality [{spec.name}]")
        formality_scores = self._formality_scorer.score(predictions)
        tqdm.write(f"Scoring BERTScore [{spec.name}]")
        bertscore = self._bertscore_scorer.score(predictions, references)
        tqdm.write(f"Scoring fluency [{spec.name}]")
        fluency_scores = self._fluency_scorer.score(predictions)

        per_example = []
        for index, record in enumerate(records):
            per_example.append(
                {
                    "source": sources[index],
                    "reference": references[index],
                    "prediction": predictions[index],
                    "metrics": {
                        "toxicity_score": toxicity_scores[index],
                        "formality_score": formality_scores[index],
                        "bert_score_precision": bertscore["precision"][index],
                        "bert_score_recall": bertscore["recall"][index],
                        "bert_score_f1": bertscore["f1"][index],
                        "bart_fluency_score": fluency_scores[index],
                    },
                }
            )

        aggregate = {
            "mean_toxicity_score": _safe_mean(toxicity_scores),
            "mean_formality_score": _safe_mean(formality_scores),
            "mean_bert_score_precision": _safe_mean(bertscore["precision"]),
            "mean_bert_score_recall": _safe_mean(bertscore["recall"]),
            "mean_bert_score_f1": _safe_mean(bertscore["f1"]),
            "mean_bart_fluency_score": _safe_mean(fluency_scores),
        }

        return {
            "system": spec.name,
            "model_name_or_path": spec.model_name_or_path,
            "aggregate_metrics": aggregate,
            "examples": per_example,
        }


def load_parallel_dataset(input_csv: Path, limit: Optional[int] = None) -> pd.DataFrame:
    dataset = pd.read_csv(input_csv)
    required = {"source", "target"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if limit is not None:
        dataset = dataset.head(limit)
    return dataset[["source", "target"]].copy()


def _build_rewriter(model_name_or_path: str) -> Seq2SeqRewriter:
    path = Path(model_name_or_path)
    if path.exists():
        return Seq2SeqRewriter(checkpoint_path=str(path))
    return Seq2SeqRewriter(model_name=model_name_or_path)


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))
