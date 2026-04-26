# Toxic-to-Neutral Style Transfer

This repo implements the same two-stage architecture described in the project slides:

1. Detect toxicity with an LLM.
2. Rewrite toxic text into neutral, respectful language with a Seq2Seq model.

## Architecture

The current production flow is:

```text
Input text
  -> OpenAI / ChatGPT toxicity detector
  -> if toxic enough:
       Seq2Seq detoxification rewriter
  -> output text
```

There are two main uses of that architecture in this repo:

1. **FastAPI inference service**
   - accepts one string or a list of strings
   - classifies toxicity
   - rewrites only when the toxicity score crosses a threshold

2. **Offline rewrite evaluation / model comparison**
   - runs detoxification models directly on a parallel dataset
   - computes the evaluation metrics discussed in the slides
   - writes comparison files for baseline vs custom models

The detection stage returns structured output with:

- `is_toxic`: binary toxic vs. non-toxic label
- `toxicity_score`: model-estimated score from 0 to 1
- `toxic_spans`: phrases that make the input toxic
- `categories`: toxicity categories such as insult, threat, profanity, harassment, hate, sexual, self-harm, or other
- `rationale`: short explanation for debugging/evaluation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already created the environment before the API/rewrite changes, rerun the install command so tokenizer dependencies such as `protobuf` and `sentencepiece` are present.

This installs the package in editable mode, so both of these work:

```bash
python -m toxic_style_transfer.classify --provider heuristic "you are an idiot"
classify-toxicity --provider heuristic "you are an idiot"
```

To use the OpenAI-backed classifier, you need an OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or put it in a local `.env` file:

```bash
OPENAI_API_KEY="your_api_key_here"
```

You can optionally choose a model:

```bash
export TOXICITY_MODEL="gpt-4o-mini"
```

`gpt-4o-mini` is the default because OpenAI's structured output docs list GPT-4o-era models and later as supporting schema-following responses.

## Repo Layout

```text
documents/      Project presentation and reference material
data/           Small checked-in examples and local dataset CSVs
experiments/    Small experiment entry points
models/         Optional trained Seq2Seq checkpoints
outputs/        Evaluation reports and comparisons
src/            Production package
tests/          Unit tests
```

## Usage

Classify with the OpenAI API:

```bash
python -m toxic_style_transfer.classify "you are an idiot"
```

Run without an API key using the local heuristic fallback:

```bash
python -m toxic_style_transfer.classify --provider heuristic "you are an idiot"
```

Classify from a file, one line at a time:

```bash
python -m toxic_style_transfer.classify --input-file data/examples.txt --provider heuristic
```

Run the API:

```bash
toxicity-api
```

or

```bash
uvicorn toxic_style_transfer.api:app --host 127.0.0.1 --port 8000
```

Example request with one string:

```bash
curl -X POST http://127.0.0.1:8000/transform \
  -H "Content-Type: application/json" \
  -d '{"text":"you are an idiot"}'
```

Example request with an array:

```bash
curl -X POST http://127.0.0.1:8000/transform \
  -H "Content-Type: application/json" \
  -d '{"text":["you are an idiot","thanks for the update"],"toxicity_threshold":0.5}'
```

The response includes per-item metadata plus `output_text` for single input or `output_texts` for batch input.

## Rewriter Model

The rewrite stage uses a Seq2Seq transformer. By default it loads `Ribin/t5-base_detoxParaphraser`, a T5 model fine-tuned for toxic-to-neutral rewriting on ParaDetox-style data.

If you train your own checkpoint, point the API at it with:

```bash
export REWRITE_MODEL_PATH=models/rewriter
```

Train a custom checkpoint from a CSV with `source,target` columns:

```bash
train-rewriter --input-csv data/paradetox_train.csv --output-dir models/rewriter
```

## Evaluation

The slide deck's rewrite evaluation rubric is implemented for direct Seq2Seq model comparison.

For the real project workflow, use **ParaDetox** from the slides.

Prepare local train/eval/test CSV files from Hugging Face:

```bash
prepare-paradetox --output-dir data/paradetox
```

That creates:

- `data/paradetox/paradetox_train.csv`
- `data/paradetox/paradetox_eval.csv`
- `data/paradetox/paradetox_test.csv`

The evaluator currently computes:

- Toxicity Score with `Detoxify`
- Professional/Formality Score with `s-nlp/roberta-base-formality-ranker`
- Meaning Preservation with `BERTScore`
- Fluency with a BARTScore-style sequence likelihood proxy

By default the BERTScore implementation uses `roberta-base`, and the fluency proxy uses `facebook/bart-base`, which keeps evaluation practical on a laptop.

Use a held-out CSV with `source,target` columns. A tiny sample file lives at `data/sample_parallel_eval.csv`, but the intended real dataset is `data/paradetox/paradetox_test.csv`.

Evaluate the current baseline model:

```bash
evaluate-rewriters --input-csv data/sample_parallel_eval.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser
```

Evaluate the real baseline on ParaDetox test data:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --output outputs/evaluations/paradetox_baseline_report.json
```

For a quick smoke test before a long run:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --limit 50 \
  --output outputs/evaluations/paradetox_baseline_report_small.json
```

Compare the baseline against your future custom checkpoint in one run:

```bash
evaluate-rewriters --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --system custom=models/rewriter
```

This writes:

- a detailed JSON report with per-example outputs and metrics
- a summary CSV with one row per system for later comparison

Example comparison command after training your own model:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --system custom=models/rewriter \
  --output outputs/evaluations/paradetox_baseline_vs_custom.json
```

Expected comparison files:

- `outputs/evaluations/paradetox_baseline_vs_custom.json`
- `outputs/evaluations/paradetox_baseline_vs_custom_summary.csv`

## Progress Bars

The evaluator prints terminal progress bars so long runs are less mysterious.

- `Systems`
  - how many rewrite systems are being evaluated
  - `1/1` means one model total
  - `1/2` means it finished the first of two models

- `Rewriting [baseline]`
  - how many dataset rows have been rewritten by that system
  - for example `15/1975` means 15 out of 1975 examples are done for the `baseline` model

After rewriting finishes for a system, the evaluator prints stage messages for the scoring passes:

- toxicity
- formality
- BERTScore
- fluency

## Recommended Workflow

1. Prepare ParaDetox locally:

```bash
prepare-paradetox --output-dir data/paradetox
```

2. Run a small baseline evaluation:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --limit 50 \
  --output outputs/evaluations/paradetox_baseline_report_small.json
```

3. Run the full baseline evaluation:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --output outputs/evaluations/paradetox_baseline_report.json
```

4. Train your own Seq2Seq model:

```bash
train-rewriter \
  --input-csv data/paradetox/paradetox_train.csv \
  --output-dir models/rewriter
```

5. Compare baseline vs custom:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --system custom=models/rewriter \
  --output outputs/evaluations/paradetox_baseline_vs_custom.json
```

## Notes

- The FastAPI app uses the full detector -> rewriter pipeline.
- The offline evaluator compares rewrite systems directly on parallel toxic/neutral pairs.
- The first large evaluation run can take a long time because it may download multiple models and score thousands of examples.
- Downloaded caches live under `.model_cache/` and are ignored by Git.

## TODO: Project Goal From Slides

The slides describe the target project as:

1. LLM-based toxicity detection
2. **Custom Seq2Seq rewriting model**
3. Evaluation comparing systems on detoxification quality

The current repo already covers:

- OpenAI-based toxicity detection
- a baseline public detoxification model
- evaluation infrastructure

The next project milestone is to train and compare **your own Seq2Seq detoxification model**.

### TODO 1: Train a new custom Seq2Seq model

Train your own checkpoint on ParaDetox:

```bash
train-rewriter \
  --input-csv data/paradetox/paradetox_train.csv \
  --output-dir models/rewriter_custom
```

You can later change hyperparameters such as:

- `--model-name`
- `--batch-size`
- `--epochs`
- `--learning-rate`

Example:

```bash
train-rewriter \
  --input-csv data/paradetox/paradetox_train.csv \
  --output-dir models/rewriter_custom \
  --model-name google/flan-t5-small \
  --epochs 5 \
  --batch-size 4
```

### TODO 2: Update the inference pipeline to use the custom model

To make the API use your custom trained model, set:

```bash
export REWRITE_MODEL_PATH=models/rewriter_custom
```

Then restart the API:

```bash
pkill -f uvicorn
toxicity-api
```

After that, `/transform` will still use the same architecture, but the rewrite stage will use your trained checkpoint instead of the baseline public model.

### TODO 3: Evaluate the custom model

Run evaluation on the custom checkpoint alone:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system custom=models/rewriter_custom \
  --output outputs/evaluations/paradetox_custom_report.json
```

### TODO 4: Compare baseline vs custom

This is the main comparison command you will want for the final project:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --system custom=models/rewriter_custom \
  --output outputs/evaluations/paradetox_baseline_vs_custom.json
```

This produces:

- `outputs/evaluations/paradetox_baseline_vs_custom.json`
- `outputs/evaluations/paradetox_baseline_vs_custom_summary.csv`

### How the commands change

Before training your own model:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --output outputs/evaluations/paradetox_baseline_report.json
```

After training your own model:

```bash
evaluate-rewriters \
  --input-csv data/paradetox/paradetox_test.csv \
  --system baseline=Ribin/t5-base_detoxParaphraser \
  --system custom=models/rewriter_custom \
  --output outputs/evaluations/paradetox_baseline_vs_custom.json
```

And for the live API, the only change is:

```bash
export REWRITE_MODEL_PATH=models/rewriter_custom
```

so the same endpoint starts using your trained model.

## Current Baseline Results

The baseline public detoxification model has already been evaluated on:

- `data/paradetox/paradetox_test.csv`

Report files:

- [paradetox_baseline_report.json](</Users/govindsinghal/Documents/Spring 2026 Classes/MSML641/nlp_style_transfer/outputs/evaluations/paradetox_baseline_report.json>)
- [paradetox_baseline_report_summary.csv](</Users/govindsinghal/Documents/Spring 2026 Classes/MSML641/nlp_style_transfer/outputs/evaluations/paradetox_baseline_report_summary.csv>)

Baseline system:

- `Ribin/t5-base_detoxParaphraser`

Aggregate results on 1975 test examples:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Mean Toxicity Score | `0.0908` | Lower is better. The model usually removes toxic wording successfully. |
| Mean Formality Score | `0.3938` | Higher is better. Detoxification is stronger than professionalism/style improvement. |
| Mean BERTScore F1 | `0.7498` | Higher is better. Meaning preservation is decent but not perfect. |
| Mean BART Fluency Score | `-6.1919` | Less negative is better. Fluency is usable but inconsistent. |

Practical takeaway:

- the baseline is a good starting point for **toxicity reduction**
- it is only moderate at **formal/professional rewriting**
- it preserves meaning reasonably well, but not perfectly
- it gives a solid comparison target for the custom Seq2Seq model from the project plan

When comparing your future custom model against this baseline, the main goal is:

- keep toxicity at or below `0.0908`
- improve formality above `0.3938`
- improve meaning preservation above `0.7498` BERTScore F1
- improve fluency to be less negative than `-6.1919`
