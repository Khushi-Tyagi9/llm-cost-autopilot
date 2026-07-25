"""
End-to-end test: classify a prompt, route it, verify (by sample rate),
and log everything to SQLite. This is the core loop your API will
eventually wrap in Phase 5.
"""
import joblib
import numpy as np

from src.classifier.features import featurize
from src.router.config_loader import load_routing_config, resolve_model_for_tier
from src.models.router_client import send_request
from src.verification.judge import verify_response
from src.verification.scheduler import should_verify
from src.logging.db import init_db, log_request

# Load trained classifier
model_bundle = joblib.load("src/classifier/model.pkl")
clf = model_bundle["model"]
scaler = model_bundle["scaler"]
needs_scaler = model_bundle["needs_scaler"]

# Load routing config
config = load_routing_config()
sample_rate = config["verification"]["sample_rate"]
thresholds = config["verification"]["divergence_threshold"]

init_db()


def predict_tier(prompt: str) -> int:
    X = np.array(featurize([prompt]))
    if needs_scaler:
        X = scaler.transform(X)
    return int(clf.predict(X)[0])


def handle_request(prompt: str):
    tier = predict_tier(prompt)
    model_config = resolve_model_for_tier(tier, config)

    response = send_request(prompt, model_config)
    print(f"\nPROMPT: {prompt}")
    print(f"  Predicted tier: {tier} -> {response.provider}/{response.model_id}")
    print(f"  Cost: ${response.cost:.6f} | Latency: {response.latency:.2f}s")

    verification = None
    if should_verify(sample_rate):
        from src.models.config import GROQ_PREMIUM
        threshold = thresholds[tier]
        verification = verify_response(prompt, response.text, GROQ_PREMIUM, threshold)
        print(f"  Verified - similarity: {verification['similarity']:.2f} | escalate: {verification['escalate']}")

    log_request(tier, response, verification, prompt)
    return response, verification


if __name__ == "__main__":
    test_prompts = [
        "What is the capital of Spain?",
        "Summarize the key benefits of remote work.",
        "Design a rate-limiting system for a public API.",
        "Extract the date from: 'The event is on June 5, 2026.'",
        "Write a short poem about the ocean.",
    ]

    for prompt in test_prompts:
        handle_request(prompt)

    print("\nAll requests logged to data/requests.db")