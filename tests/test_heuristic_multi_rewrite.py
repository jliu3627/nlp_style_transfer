from toxic_style_transfer.classifier import ToxicityClassifier
from toxic_style_transfer.rewriter import (
    Seq2SeqRewriter,
    _classification_for_segment,
    _split_text_segments,
)


def test_heuristic_classifier_rewriter_smoke():
    text = (
        "I read through the proposal this morning, and most of the structure makes sense. "
        "Calling the analyst an idiot during review made the feedback harder to use. "
        "The timeline risk still needs a mitigation plan before Friday. "
        "That budget argument was stupid, even though the underlying concern is valid. "
        "Please keep the revised cost assumptions in the next draft. "
        "Saying the milestone plan is useless does not help the team understand the actual blocker. "
        "Thanks for sending the update, and I can review the revised numbers after lunch."
    )

    print(f"attempting to classify {text}")
    segments = _split_text_segments(text)
    classification = ToxicityClassifier(provider="heuristic").classify(text)

    print("sentence segments:")
    for index, (segment_text, start, end) in enumerate(segments, start=1):
        print(f"  {index}. [{start}:{end}] {segment_text}")
        segment_classification = _classification_for_segment(
            segment_text,
            start,
            end,
            classification,
        )
        toxic_spans = (
            [span.to_dict() for span in segment_classification.toxic_spans]
            if segment_classification
            else []
        )
        print(f"     toxic={segment_classification is not None}; spans={toxic_spans}")

    print("classification =", classification.to_dict())
    rewritten = Seq2SeqRewriter().rewrite(text, classification)
    print("rewritten =", rewritten)

    assert classification.is_toxic
    assert len(classification.toxic_spans) >= 3
    assert len(segments) == 7
    assert "idiot" not in rewritten.lower()
    assert "stupid" not in rewritten.lower()
    assert rewritten != text
    assert "The timeline risk still needs a mitigation plan before Friday." in rewritten
    assert "Please keep the revised cost assumptions in the next draft." in rewritten
    assert "Thanks for sending the update" in rewritten
