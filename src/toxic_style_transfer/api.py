"""FastAPI service for the toxic-to-neutral pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from toxic_style_transfer.pipeline import ToxicityRewritePipeline

FRONTEND_PATH = Path(__file__).with_name("index.html")


def _frontend_html() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


class TransformRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="A single string or a list of strings.")
    toxicity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rewrite only when the toxicity score meets or exceeds this value.",
    )


class ToxicSpanResponse(BaseModel):
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    category: str


class ClassificationResponse(BaseModel):
    text: str
    is_toxic: bool
    toxicity_score: float
    toxic_spans: List[ToxicSpanResponse]
    categories: List[str]
    rationale: str


class TransformResultResponse(BaseModel):
    input_text: str
    output_text: str
    rewritten: bool
    classification: ClassificationResponse


class TransformResponse(BaseModel):
    results: List[TransformResultResponse]
    output_text: Optional[str] = None
    output_texts: List[str]


def create_app(pipeline: Optional[ToxicityRewritePipeline] = None) -> FastAPI:
    app = FastAPI(title="Toxic-to-Neutral API", version="0.2.0")
    app.state.pipeline = pipeline
    app.state.cached_pipeline = None

    @app.get("/", response_class=HTMLResponse)
    def frontend() -> HTMLResponse:
        return HTMLResponse(_frontend_html())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transform", response_model=TransformResponse)
    def transform(request: TransformRequest) -> TransformResponse:
        texts = [request.text] if isinstance(request.text, str) else request.text
        active_pipeline = app.state.pipeline
        if active_pipeline is None:
            if app.state.cached_pipeline is None:
                app.state.cached_pipeline = _build_pipeline()
            active_pipeline = app.state.cached_pipeline
        results = [
            item.to_dict()
            for item in active_pipeline.transform_many(
                texts,
                toxicity_threshold=request.toxicity_threshold,
            )
        ]
        output_texts = [item["output_text"] for item in results]
        return TransformResponse(
            results=results,
            output_text=output_texts[0] if len(output_texts) == 1 else None,
            output_texts=output_texts,
        )

    return app


def _build_pipeline() -> ToxicityRewritePipeline:
    threshold = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))
    provider = os.getenv("TOXICITY_PROVIDER", "openai")
    return ToxicityRewritePipeline(
        classifier_provider=provider,
        toxicity_threshold=threshold,
    )


def run() -> None:
    load_dotenv()
    import uvicorn

    uvicorn.run(
        "toxic_style_transfer.api:create_app",
        factory=True,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


app = create_app()
