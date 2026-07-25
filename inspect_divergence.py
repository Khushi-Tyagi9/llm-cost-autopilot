from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request
from src.verification.judge import verify_response

prompt = "Extract the date from: 'The event is on June 5, 2026.'"

cheap = send_request(prompt, GROQ_CHEAP)
print("CHEAP MODEL OUTPUT:")
print(repr(cheap.text))

result = verify_response(prompt, cheap.text, GROQ_PREMIUM, divergence_threshold=0.20)
print("\nPREMIUM MODEL OUTPUT:")
print(repr(result["premium_text"]))

print(f"\nSimilarity: {result['similarity']:.2f}")
print(f"Divergence: {result['divergence']:.2f}")
print(f"Escalate: {result['escalate']}")