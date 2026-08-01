"""
Loads routing.yaml and resolves tier -> ModelConfig.

Builds ModelConfig objects directly from the YAML's model definitions,
so adding a new model (any provider, any pricing) only requires editing
routing.yaml - no Python changes needed.
"""
import yaml

from src.models.config import ModelConfig


def load_routing_config(path="config/routing.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_configs(routing_config: dict) -> dict:
    """Returns {name: ModelConfig} built directly from the YAML definitions."""
    configs = {}
    for name, spec in routing_config["models"].items():
        configs[name] = ModelConfig(
            provider=spec["provider"],
            model_id=spec["model_id"],
            cost_per_input_token=spec["cost_per_input_token"],
            cost_per_output_token=spec["cost_per_output_token"],
            quality_tier=spec.get("quality_tier", 0),
        )
    return configs


def resolve_model_for_tier(tier: int, routing_config: dict) -> ModelConfig:
    model_name = routing_config["routing"][tier]["model"]
    model_configs = build_model_configs(routing_config)
    return model_configs[model_name]