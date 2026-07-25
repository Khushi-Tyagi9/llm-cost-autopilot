from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request
from src.verification.judge import verify_response
from src.verification.scheduler import should_verify
from src.router.config_loader import load_routing_config

config = load_routing_config()
sample_rate = config["verification"]["sample_rate"]
thresholds = config["verification"]["divergence_threshold"]

# (tier, prompt) pairs so we use the right threshold per tier
test_cases = [
    (1, "What is the capital of France?"),
    (3, "Write a short story about a robot discovering emotions."),
    (2, "Summarize the differences between SQL and NoSQL databases."),
]

for tier, prompt in test_cases:
    threshold = thresholds[tier]
    print(f"\nPROMPT (tier {tier}, threshold {threshold}): {prompt}")
    cheap_response = send_request(prompt, GROQ_CHEAP)
    print(f"  CHEAP ANSWER: {cheap_response.text[:100]}...")

    if should_verify(sample_rate):
        result = verify_response(prompt, cheap_response.text, GROQ_PREMIUM, threshold)
        print(f"  Similarity: {result['similarity']:.2f} | Divergence: {result['divergence']:.2f}")
        print(f"  ESCALATE: {result['escalate']}")
    else:
        print("  (skipped verification this time - sampling rate)")