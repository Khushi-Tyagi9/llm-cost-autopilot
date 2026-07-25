"""
Loads routing.yaml and resolves tier -> ModelConfig.
"""
import yaml

from src.models.config import ModelConfig


def load_routing_config(path="config/routing.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_configs(routing_config: dict) -> dict:
    """Returns {name: ModelConfig} for every model defined in the YAML."""
    configs = {}
    for name, spec in routing_config["models"].items():
        # Pricing lives in src/models/config.py's constants; look them up
        # by matching model_id so we don't duplicate pricing in two places.
        from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
        known = {GROQ_CHEAP.model_id: GROQ_CHEAP, GROQ_PREMIUM.model_id: GROQ_PREMIUM}
        configs[name] = known[spec["model_id"]]
    return configs


def resolve_model_for_tier(tier: int, routing_config: dict) -> ModelConfig:
    model_name = routing_config["routing"][tier]["model"]
    model_configs = build_model_configs(routing_config)
    return model_configs[model_name]