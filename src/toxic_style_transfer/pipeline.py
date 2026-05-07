"""End-to-end toxicity detection and rewriting pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Protocol

from toxic_style_transfer.classifier import ToxicityClassifier
from toxic_style_transfer.rewriter import Seq2SeqRewriter
from toxic_style_transfer.schemas import TextTransformation, ToxicityClassification


class RewriterClient(Protocol):
    def rewrite(
        self,
        text: str,
        classification: Optional[ToxicityClassification] = None,
    ) -> str:
        """Return a neutral rewrite of the input text."""


class ToxicityRewritePipeline:
    """Project architecture: LLM detection followed by Seq2Seq rewriting."""

    def __init__(
        self,
        classifier: Optional[ToxicityClassifier] = None,
        rewriter: Optional[RewriterClient] = None,
        classifier_provider: str = "openai",
        toxicity_threshold: float = 0.5,
    ) -> None:
        self._classifier = classifier or ToxicityClassifier(provider=classifier_provider)
        self._rewriter = rewriter or Seq2SeqRewriter()
        self._toxicity_threshold = toxicity_threshold

    def transform_text(
        self,
        text: str,
        toxicity_threshold: Optional[float] = None,
    ) -> TextTransformation:
        classification = self._classifier.classify(text)
        threshold = self._toxicity_threshold if toxicity_threshold is None else toxicity_threshold
        should_rewrite = classification.is_toxic and classification.toxicity_score >= threshold
        output_text = (
            self._rewriter.rewrite(text, classification)
            if should_rewrite
            else text
        )
        return TextTransformation(
            input_text=text,
            output_text=output_text,
            classification=classification,
            rewritten=should_rewrite and output_text != text,
        )

    def transform_many(
        self,
        texts: Iterable[str],
        toxicity_threshold: Optional[float] = None,
    ) -> List[TextTransformation]:
        return [
            self.transform_text(text, toxicity_threshold=toxicity_threshold)
            for text in texts
        ]
