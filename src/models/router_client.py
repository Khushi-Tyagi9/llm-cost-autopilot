from dotenv import load_dotenv
from src.models.config import ModelConfig
from src.models.response import Response
from src.models.providers.groq_provider import call_groq
from src.models.providers.claude_provider import call_claude
from src.models.providers.openai_provider import call_openai
import time as time_module
from groq import RateLimitError

load_dotenv()


def _call_provider(prompt: str, model_config: ModelConfig):
    if model_config.provider == "groq":
        return call_groq(prompt, model_config.model_id)
    elif model_config.provider == "anthropic":
        return call_claude(prompt, model_config.model_id)
    elif model_config.provider == "openai":
        return call_openai(prompt, model_config.model_id)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")


def send_request(prompt: str, model_config: ModelConfig, max_retries: int = 2) -> Response:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            text, input_tokens, output_tokens, latency = _call_provider(prompt, model_config)
            break
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                time_module.sleep(2 ** attempt)  # 1s, 2s backoff
                continue
            raise

    cost = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )

    return Response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        cost=cost,
        model_id=model_config.model_id,
        provider=model_config.provider,
    )