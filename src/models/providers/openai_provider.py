"""
Thin wrapper around the OpenAI SDK. Mirrors groq_provider.py and
claude_provider.py's shape exactly so router_client.py can treat all
three providers identically.
"""
import os
import time

from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client


def call_openai(prompt: str, model_id: str) -> tuple[str, int, int, float]:
    client = _get_client()

    start = time.monotonic()
    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.monotonic() - start

    text = completion.choices[0].message.content
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    return text, input_tokens, output_tokens, latency
