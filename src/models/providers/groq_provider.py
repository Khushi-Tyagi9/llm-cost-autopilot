import os
import time

from groq import Groq

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        _client = Groq(api_key=api_key)
    return _client


def call_groq(prompt: str, model_id: str):
    client = _get_client()

    start = time.monotonic()
    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.monotonic() - start

    text = completion.choices[0].message.content
    usage = completion.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    return text, input_tokens, output_tokens, latency