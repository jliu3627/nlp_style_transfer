"""Provider-neutral classifier facade."""

from __future__ import annotations

from typing import Optional, Protocol

from toxic_style_transfer.heuristic import HeuristicToxicityClient
from toxic_style_transfer.openai_client import OpenAIToxicityClient
from toxic_style_transfer.schemas import ToxicityClassification


class ToxicityClient(Protocol):
    def classify(self, text: str) -> ToxicityClassification:
        """Return structured toxicity classification for one text."""


class ToxicityClassifier:
    """First pipeline stage: toxic vs. non-toxic classification."""

    def __init__(self, client: Optional[ToxicityClient] = None, provider: str = "openai") -> None:
        self._client = client or _client_for_provider(provider)

    def classify(self, text: str) -> ToxicityClassification:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Cannot classify empty text.")
        return self._client.classify(normalized)


def _client_for_provider(provider: str) -> ToxicityClient:
    if provider == "openai":
        return OpenAIToxicityClient()
    if provider == "heuristic":
        return HeuristicToxicityClient()
    raise ValueError(f"Unsupported provider: {provider}")
