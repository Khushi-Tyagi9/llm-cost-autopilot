from dotenv import load_dotenv

from src.models.config import ModelConfig
from src.models.response import Response
from src.models.providers.groq_provider import call_groq

load_dotenv()


def send_request(prompt: str, model_config: ModelConfig) -> Response:
    if model_config.provider == "groq":
        text, input_tokens, output_tokens, latency = call_groq(prompt, model_config.model_id)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")

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