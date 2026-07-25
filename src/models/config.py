from dataclasses import dataclass


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    quality_tier: int
    latency_estimate: float = 0.0


GROQ_CHEAP = ModelConfig(
    provider="groq",
    model_id="llama-3.1-8b-instant",
    cost_per_input_token=0.05 / 1_000_000,
    cost_per_output_token=0.08 / 1_000_000,
    quality_tier=1,
)

GROQ_PREMIUM = ModelConfig(
    provider="groq",
    model_id="llama-3.3-70b-versatile",
    cost_per_input_token=0.59 / 1_000_000,
    cost_per_output_token=0.79 / 1_000_000,
    quality_tier=3,
)