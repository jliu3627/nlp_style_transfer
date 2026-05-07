"""Shared data structures for toxicity detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


TOXICITY_CATEGORIES = (
    "insult",
    "threat",
    "profanity",
    "harassment",
    "hate",
    "sexual",
    "self_harm",
    "other",
)


@dataclass(frozen=True)
class ToxicSpan:
    """A phrase or sentence fragment that contributes to toxicity."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    category: str = "other"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToxicSpan":
        category = str(data.get("category") or "other")
        if category not in TOXICITY_CATEGORIES:
            category = "other"

        return cls(
            text=str(data.get("text") or ""),
            start=_optional_int(data.get("start")),
            end=_optional_int(data.get("end")),
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "category": self.category,
        }


@dataclass(frozen=True)
class ToxicityClassification:
    """Structured output for the first pipeline stage."""

    text: str
    is_toxic: bool
    toxicity_score: float
    toxic_spans: list[ToxicSpan] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_dict(cls, text: str, data: dict[str, Any]) -> "ToxicityClassification":
        spans = [
            ToxicSpan.from_dict(item)
            for item in data.get("toxic_spans", [])
            if isinstance(item, dict)
        ]
        spans = [_normalize_span_offsets(text, span) for span in spans]
        categories = _valid_categories(data.get("categories", []))
        if not categories:
            categories = sorted({span.category for span in spans if span.category != "other"})

        return cls(
            text=text,
            is_toxic=bool(data.get("is_toxic", False)),
            toxicity_score=_clamp_score(data.get("toxicity_score", 0.0)),
            toxic_spans=spans,
            categories=categories,
            rationale=str(data.get("rationale") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "is_toxic": self.is_toxic,
            "toxicity_score": self.toxicity_score,
            "toxic_spans": [span.to_dict() for span in self.toxic_spans],
            "categories": self.categories,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TextTransformation:
    """Output of the full detect-and-rewrite pipeline."""

    input_text: str
    output_text: str
    classification: ToxicityClassification
    rewritten: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "rewritten": self.rewritten,
            "classification": self.classification.to_dict(),
        }


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_span_offsets(text: str, span: ToxicSpan) -> ToxicSpan:
    if _span_offsets_match(text, span):
        return span

    if not span.text:
        return ToxicSpan(text=span.text, start=None, end=None, category=span.category)

    start = text.lower().find(span.text.lower())
    if start < 0:
        return ToxicSpan(text=span.text, start=None, end=None, category=span.category)

    end = start + len(span.text)
    return ToxicSpan(
        text=text[start:end],
        start=start,
        end=end,
        category=span.category,
    )


def _span_offsets_match(text: str, span: ToxicSpan) -> bool:
    if span.start is None or span.end is None:
        return False
    if span.start < 0 or span.end <= span.start or span.end > len(text):
        return False
    return text[span.start:span.end].lower() == span.text.lower()


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _valid_categories(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in TOXICITY_CATEGORIES]
