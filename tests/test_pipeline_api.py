from fastapi.testclient import TestClient

from toxic_style_transfer.api import create_app
from toxic_style_transfer.pipeline import ToxicityRewritePipeline
from toxic_style_transfer.schemas import ToxicityClassification


class FakeClassifier:
    def classify(self, text: str) -> ToxicityClassification:
        lowered = text.lower()
        if "idiot" in lowered:
            return ToxicityClassification(
                text=text,
                is_toxic=True,
                toxicity_score=0.91,
                categories=["insult"],
                rationale="contains insult",
            )
        return ToxicityClassification(
            text=text,
            is_toxic=False,
            toxicity_score=0.08,
            categories=[],
            rationale="clean",
        )


class FakeRewriter:
    def rewrite(self, text: str, classification=None) -> str:
        return text.replace("idiot", "person")


def test_pipeline_only_rewrites_toxic_text():
    pipeline = ToxicityRewritePipeline(
        classifier=FakeClassifier(),
        rewriter=FakeRewriter(),
        toxicity_threshold=0.5,
    )

    toxic = pipeline.transform_text("You are an idiot")
    clean = pipeline.transform_text("I disagree with you")

    assert toxic.rewritten is True
    assert toxic.output_text == "You are an person"
    assert clean.rewritten is False
    assert clean.output_text == "I disagree with you"


def test_api_accepts_single_string():
    app = create_app(
        ToxicityRewritePipeline(
            classifier=FakeClassifier(),
            rewriter=FakeRewriter(),
            toxicity_threshold=0.5,
        )
    )
    client = TestClient(app)

    response = client.post("/transform", json={"text": "You are an idiot"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["output_text"] == "You are an person"
    assert payload["output_texts"] == ["You are an person"]
    assert payload["results"][0]["classification"]["is_toxic"] is True


def test_api_accepts_array_of_strings():
    app = create_app(
        ToxicityRewritePipeline(
            classifier=FakeClassifier(),
            rewriter=FakeRewriter(),
            toxicity_threshold=0.5,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/transform",
        json={"text": ["You are an idiot", "I disagree with you"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output_text"] is None
    assert payload["output_texts"] == ["You are an person", "I disagree with you"]
    assert payload["results"][1]["rewritten"] is False
