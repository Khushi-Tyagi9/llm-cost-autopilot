"""
Thin wrapper around the Anthropic SDK. Mirrors groq_provider.py's shape
exactly so router_client.py can treat both providers identically.
"""
import os
import time

from anthropic import Anthropic

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        _client = Anthropic(api_key=api_key)
    return _client


def call_claude(prompt: str, model_id: str, max_tokens: int = 1024) -> tuple[str, int, int, float]:
    client = _get_client()

    start = time.monotonic()
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.monotonic() - start

    text = "".join(block.text for block in message.content if block.type == "text")
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    return text, input_tokens, output_tokens, latency