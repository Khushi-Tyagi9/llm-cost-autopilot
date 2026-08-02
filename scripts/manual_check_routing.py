from src.router.config_loader import load_routing_config, resolve_model_for_tier

config = load_routing_config()

for tier in [1, 2, 3]:
    model = resolve_model_for_tier(tier, config)
    print(f"Tier {tier} -> {model.provider}/{model.model_id}")