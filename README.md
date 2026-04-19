# Toxic-to-Neutral Style Transfer

This project follows the two-stage pipeline from the mid-semester presentation:

1. Detect toxicity with an LLM.
2. Rewrite toxic text into neutral, respectful language with a custom Seq2Seq model.

The first implemented stage is the LLM toxicity classifier. It returns structured output with:

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

## Project Structure

```text
documents/      Project presentation and reference material
experiments/    Training sketches and exploratory model work
src/            Production pipeline package
tests/          Local unit tests
```

