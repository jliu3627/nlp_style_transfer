"""OpenAI-backed implementation of the toxicity classifier."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv

from toxic_style_transfer.prompts import SYSTEM_PROMPT, user_prompt
from toxic_style_transfer.schemas import TOXICITY_CATEGORIES, ToxicityClassification


DEFAULT_MODEL = "gpt-4o-mini"


TOXICITY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "toxicity_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_toxic": {"type": "boolean"},
            "toxicity_score": {"type": "number"},
            "toxic_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "end": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "category": {"type": "string", "enum": list(TOXICITY_CATEGORIES)},
                    },
                    "required": ["text", "start", "end", "category"],
                    "additionalProperties": False,
                },
            },
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": list(TOXICITY_CATEGORIES)},
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "is_toxic",
            "toxicity_score",
            "toxic_spans",
            "categories",
            "rationale",
        ],
        "additionalProperties": False,
    },
}


class OpenAIToxicityClient:
    """Classifies text through OpenAI's Responses API."""

    def __init__(self, model: Optional[str] = None) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before using the OpenAI provider, "
                "or run with --provider heuristic for a local smoke test."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.getenv("TOXICITY_MODEL") or DEFAULT_MODEL

    def classify(self, text: str) -> ToxicityClassification:
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(text)},
            ],
            text={"format": TOXICITY_RESPONSE_FORMAT},
        )
        payload = _extract_json(response)
        return ToxicityClassification.from_dict(text=text, data=payload)


def _extract_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return json.loads(text)

    raise RuntimeError("OpenAI response did not contain parseable output text.")
