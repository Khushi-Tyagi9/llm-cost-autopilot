"""
Tests for src/router/config_loader.py

Covers routing config parsing and tier -> ModelConfig resolution.
Uses a temporary YAML file so tests don't depend on the real
config/routing.yaml and can't be broken by editing it, or vice versa.
"""
import pytest
import yaml
import tempfile
import os

from src.router.config_loader import (
    load_routing_config,
    build_model_configs,
    resolve_model_for_tier,
)


@pytest.fixture
def sample_config():
    return {
        "routing": {
            1: {"model": "cheap_model", "description": "simple tasks"},
            2: {"model": "cheap_model", "description": "moderate tasks"},
            3: {"model": "premium_model", "description": "complex tasks"},
        },
        "verification": {
            "sample_rate": 0.15,
        },
        "models": {
            "cheap_model": {
                "provider": "groq",
                "model_id": "llama-3.1-8b-instant",
                "cost_per_input_token": 0.00000005,
                "cost_per_output_token": 0.00000008,
                "quality_tier": 1,
            },
            "premium_model": {
                "provider": "groq",
                "model_id": "llama-3.3-70b-versatile",
                "cost_per_input_token": 0.00000059,
                "cost_per_output_token": 0.00000079,
                "quality_tier": 3,
            },
        },
    }


@pytest.fixture
def sample_config_file(sample_config):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        path = f.name
    yield path
    os.unlink(path)


class TestLoadRoutingConfig:
    def test_loads_valid_yaml(self, sample_config_file):
        config = load_routing_config(sample_config_file)
        assert "routing" in config
        assert "models" in config
        assert "verification" in config

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_routing_config("this/path/does/not/exist.yaml")


class TestBuildModelConfigs:
    def test_builds_a_config_per_model(self, sample_config):
        configs = build_model_configs(sample_config)
        assert set(configs.keys()) == {"cheap_model", "premium_model"}

    def test_config_fields_match_yaml(self, sample_config):
        configs = build_model_configs(sample_config)
        cheap = configs["cheap_model"]
        assert cheap.provider == "groq"
        assert cheap.model_id == "llama-3.1-8b-instant"
        assert cheap.cost_per_input_token == pytest.approx(0.00000005)
        assert cheap.quality_tier == 1

    def test_missing_quality_tier_defaults_gracefully(self, sample_config):
        config_copy = dict(sample_config)
        config_copy["models"] = {
            "cheap_model": {k: v for k, v in sample_config["models"]["cheap_model"].items() if k != "quality_tier"}
        }
        configs = build_model_configs(config_copy)
        assert configs["cheap_model"].quality_tier == 0


class TestResolveModelForTier:
    def test_tier_1_resolves_to_cheap_model(self, sample_config):
        model = resolve_model_for_tier(1, sample_config)
        assert model.model_id == "llama-3.1-8b-instant"

    def test_tier_3_resolves_to_premium_model(self, sample_config):
        model = resolve_model_for_tier(3, sample_config)
        assert model.model_id == "llama-3.3-70b-versatile"

    def test_unknown_tier_raises_key_error(self, sample_config):
        with pytest.raises(KeyError):
            resolve_model_for_tier(99, sample_config)