"""Seq2Seq text rewriter used for stage 2 of the pipeline."""

from __future__ import annotations

import os
from typing import Any, Optional
from dotenv import load_dotenv

from toxic_style_transfer.schemas import ToxicityClassification


DEFAULT_REWRITE_MODEL = "Ribin/t5-base_detoxParaphraser"
DEFAULT_MAX_NEW_TOKENS = 96


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
