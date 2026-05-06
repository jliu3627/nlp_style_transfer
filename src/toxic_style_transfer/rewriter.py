"""Seq2Seq text rewriter used for stage 2 of the pipeline."""

from __future__ import annotations

import os
from typing import Any, Optional
from dotenv import load_dotenv

from toxic_style_transfer.schemas import ToxicSpan, ToxicityClassification


DEFAULT_REWRITE_MODEL = "Ribin/t5-base_detoxParaphraser"
DEFAULT_MAX_NEW_TOKENS = 96
_SENTENCIZER: Optional[Any] = None


def _default_device_name() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Seq2SeqRewriter:
    """Rewrite toxic text into neutral text with a transformer model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        load_dotenv()
        self._model_name = (
            checkpoint_path
            or model_name
            or os.getenv("REWRITE_MODEL_PATH")
            or os.getenv("REWRITE_MODEL_NAME")
            or DEFAULT_REWRITE_MODEL
        )
        self._max_new_tokens = int(
            max_new_tokens
            or os.getenv("REWRITE_MAX_NEW_TOKENS")
            or DEFAULT_MAX_NEW_TOKENS
        )
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: Optional[Any] = None

    def rewrite(self, text: str, classification: Optional[ToxicityClassification] = None) -> str:
        if _should_split_text(text):
            return self._rewrite_multi_sentence(text, classification)
        return self._rewrite_segment(text, classification)

    def _rewrite_multi_sentence(
        self,
        text: str,
        classification: Optional[ToxicityClassification] = None,
    ) -> str:
        segments = _split_text_segments(text)
        rewritten_segments: list[str] = []
        for segment_text, start, end in segments:
            segment_classification = _classification_for_segment(
                segment_text,
                start,
                end,
                classification,
            )
            if classification and classification.is_toxic and segment_classification is None:
                rewritten_segments.append(segment_text)
                continue
            rewritten_segments.append(self._rewrite_segment(segment_text, segment_classification))
        return " ".join(segment.strip() for segment in rewritten_segments if segment.strip()) or text

    def _rewrite_segment(
        self,
        text: str,
        classification: Optional[ToxicityClassification] = None,
    ) -> str:
        prompts = _build_prompts(text, classification, self._model_name)
        for prompt in prompts:
            output_text = self._generate(prompt)
            cleaned = _clean_output(output_text)
            if cleaned and cleaned.lower() != text.strip().lower():
                return cleaned
        return text

    def _generate(self, prompt: str) -> str:
        tokenizer = self._get_tokenizer()
        model = self._get_model()
        device = self._get_device()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        torch = self._torch()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                num_beams=5,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def _get_model(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM

            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            self._model = self._model.to(self._get_device())
            self._model.eval()
        return self._model

    def _get_device(self):
        if self._device is None:
            torch = self._torch()
            self._device = torch.device(_default_device_name())
        return self._device

    @staticmethod
    def _torch():
        import torch

        return torch


def _should_split_text(text: str) -> bool:
    return len(_split_text_segments(text)) > 1


def _split_text_segments(text: str) -> list[tuple[str, int, int]]:
    sentencizer = _get_sentencizer()
    doc = sentencizer(text)
    segments = []
    for sentence in doc.sents:
        segment = sentence.text.strip()
        if segment:
            segments.append((segment, sentence.start_char, sentence.end_char))
    return segments


def _get_sentencizer():
    global _SENTENCIZER
    if _SENTENCIZER is None:
        import spacy

        _SENTENCIZER = spacy.blank("en")
        _SENTENCIZER.add_pipe("sentencizer")
    return _SENTENCIZER


def _classification_for_segment(
    segment_text: str,
    start: int,
    end: int,
    classification: Optional[ToxicityClassification],
) -> Optional[ToxicityClassification]:
    if classification is None or not classification.is_toxic:
        return classification

    if not classification.toxic_spans:
        return ToxicityClassification(
            text=segment_text,
            is_toxic=True,
            toxicity_score=classification.toxicity_score,
            toxic_spans=[],
            categories=classification.categories,
            rationale=classification.rationale,
        )

    segment_spans = []
    for span in classification.toxic_spans:
        if span.start is None or span.end is None:
            if span.text and span.text.lower() in segment_text.lower():
                segment_spans.append(
                    ToxicSpan(
                        text=span.text,
                        start=None,
                        end=None,
                        category=span.category,
                    )
                )
            continue
        if span.end <= start or span.start >= end:
            continue
        relative_start = max(span.start - start, 0)
        relative_end = min(span.end - start, len(segment_text))
        segment_spans.append(
            ToxicSpan(
                text=segment_text[relative_start:relative_end] or span.text,
                start=relative_start,
                end=relative_end,
                category=span.category,
            )
        )

    if not segment_spans:
        return None

    return ToxicityClassification(
        text=segment_text,
        is_toxic=True,
        toxicity_score=classification.toxicity_score,
        toxic_spans=segment_spans,
        categories=sorted({span.category for span in segment_spans}),
        rationale=classification.rationale,
    )


def _build_prompts(
    text: str,
    classification: Optional[ToxicityClassification],
    model_name: str,
) -> list[str]:
    prompts = []
    lowered_name = model_name.lower()
    toxic_spans = ""
    if classification and classification.toxic_spans:
        toxic_spans = ", ".join(span.text for span in classification.toxic_spans if span.text)

    if "detox" in lowered_name or "paradetox" in lowered_name:
        prompts.append(f"Toxic version: {text}")
        if toxic_spans:
            prompts.append(f"Toxic version: {text}\nToxic spans: {toxic_spans}")

    prompts.append(
        "rewrite toxic to neutral: " + text
    )
    prompts.append(
        "Rewrite the following text so it is neutral and respectful while preserving the meaning: "
        + text
    )
    if toxic_spans:
        prompts.append(
            "Rewrite the following text so it is neutral and respectful while preserving the "
            f"meaning. Replace these toxic spans: {toxic_spans}\nText: {text}"
        )
    return prompts


def _clean_output(output_text: str) -> str:
    cleaned = output_text.strip()
    prefixes = (
        "Non-toxic version:",
        "Non toxic version:",
        "Neutral version:",
        "Rewritten text:",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned
