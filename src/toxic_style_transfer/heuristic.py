"""Tiny local classifier used for tests and no-key smoke runs."""

from __future__ import annotations

import re

from toxic_style_transfer.schemas import ToxicSpan, ToxicityClassification


TOXIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bidiot\b", "insult"),
    (r"\bstupid\b", "insult"),
    (r"\buseless\b", "insult"),
    (r"\bshut up\b", "harassment"),
    (r"\bkill yourself\b", "self_harm"),
    (r"\bgo die\b", "threat"),
)


class HeuristicToxicityClient:
    """A deterministic fallback for development without API calls."""

    def classify(self, text: str) -> ToxicityClassification:
        spans: list[ToxicSpan] = []
        for pattern, category in TOXIC_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                spans.append(
                    ToxicSpan(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        category=category,
                    )
                )

        categories = sorted({span.category for span in spans})
        is_toxic = bool(spans)
        score = min(1.0, 0.35 + 0.2 * len(spans)) if is_toxic else 0.05
        rationale = (
            "Matched local toxic phrase patterns."
            if is_toxic
            else "No local toxic phrase patterns matched."
        )

        return ToxicityClassification(
            text=text,
            is_toxic=is_toxic,
            toxicity_score=score,
            toxic_spans=spans,
            categories=categories,
            rationale=rationale,
        )
