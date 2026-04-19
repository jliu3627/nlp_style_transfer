from toxic_style_transfer.classifier import ToxicityClassifier
from toxic_style_transfer.heuristic import HeuristicToxicityClient
from toxic_style_transfer.schemas import ToxicityClassification


class FakeClient:
    def classify(self, text: str) -> ToxicityClassification:
        return ToxicityClassification(
            text=text,
            is_toxic=False,
            toxicity_score=0.01,
            rationale="fake",
        )


def test_classifier_rejects_empty_text():
    classifier = ToxicityClassifier(client=FakeClient())

    try:
        classifier.classify("   ")
    except ValueError as exc:
        assert "empty text" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty text")


def test_heuristic_classifier_detects_toxic_span_offsets():
    result = HeuristicToxicityClient().classify("You are an idiot.")

    assert result.is_toxic is True
    assert result.categories == ["insult"]
    assert result.toxic_spans[0].text == "idiot"
    assert result.toxic_spans[0].start == 11
    assert result.toxic_spans[0].end == 16
