"""Prompts for LLM-based toxicity detection."""

SYSTEM_PROMPT = """You classify user text for toxicity before a later neutral rewriting step.

Toxic text includes insults, identity attacks, harassment, threats, aggressive profanity, sexual harassment, and language encouraging self-harm. Mild disagreement, criticism, or negative sentiment is not automatically toxic.

Return only the requested structured fields. Mark short spans that directly contribute to toxicity. Use character offsets from the original input when confident; otherwise use null offsets. Keep the rationale short and evaluation-focused."""


def user_prompt(text: str) -> str:
    return f"Classify this text for toxicity and toxic spans:\n\n{text}"
