from pathlib import Path

from toxic_style_transfer.evaluation import RewriteEvaluator, RewriteSystemSpec, load_parallel_dataset


class FakeToxicityScorer:
    def score(self, texts):
        return [0.1 for _ in texts]


class FakeFormalityScorer:
    def score(self, texts, batch_size=8):
        return [0.9 for _ in texts]


class FakeBERTScoreScorer:
    def score(self, candidates, references):
        size = len(candidates)
        return {
            "precision": [0.8] * size,
            "recall": [0.7] * size,
            "f1": [0.75] * size,
        }


class FakeFluencyScorer:
    def score(self, texts, batch_size=4):
        return [-1.2 for _ in texts]


class FakeRewriter:
    def __init__(self, prediction_prefix):
        self._prediction_prefix = prediction_prefix

    def rewrite(self, text, classification=None):
        return f"{self._prediction_prefix}:{text}"


def test_load_parallel_dataset_enforces_columns(tmp_path: Path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("source,target\nhello,hi\n", encoding="utf-8")

    dataset = load_parallel_dataset(csv_path)

    assert list(dataset.columns) == ["source", "target"]
    assert len(dataset) == 1


def test_evaluator_aggregates_metrics(monkeypatch):
    monkeypatch.setattr(
        "toxic_style_transfer.evaluation._build_rewriter",
        lambda model_name_or_path: FakeRewriter(model_name_or_path),
    )

    evaluator = RewriteEvaluator(
        toxicity_scorer=FakeToxicityScorer(),
        formality_scorer=FakeFormalityScorer(),
        bertscore_scorer=FakeBERTScoreScorer(),
        fluency_scorer=FakeFluencyScorer(),
    )

    dataset = load_parallel_dataset(Path("data/sample_parallel_eval.csv"), limit=2)
    report = evaluator.evaluate_systems(
        dataset=dataset,
        systems=[RewriteSystemSpec(name="baseline", model_name_or_path="fake-model")],
    )

    system = report["systems"][0]
    assert system["system"] == "baseline"
    assert system["aggregate_metrics"]["mean_toxicity_score"] == 0.1
    assert system["aggregate_metrics"]["mean_formality_score"] == 0.9
    assert system["aggregate_metrics"]["mean_bert_score_f1"] == 0.75
    assert system["examples"][0]["prediction"].startswith("fake-model:")
