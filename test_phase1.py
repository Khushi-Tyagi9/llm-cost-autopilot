from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request

test_prompts = [
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in 2 sentences.",
    "Write a haiku about autumn.",
    "Extract the email address from: 'Contact me at jane@example.com for details.'",
    "Compare REST and GraphQL in 3 bullet points.",
    "What's 15% of 240?",
    "Reformat this as a bulleted list: apples, bananas, cherries",
    "Explain quantum entanglement to a 10-year-old.",
    "Translate 'good morning' into Spanish.",
    "Write a short function signature (no implementation) for reversing a linked list.",
]

for prompt in test_prompts:
    print(f"\nPROMPT: {prompt}")
    cheap = send_request(prompt, GROQ_CHEAP)
    print("  CHEAP:  ", cheap)
    premium = send_request(prompt, GROQ_PREMIUM)
    print("  PREMIUM:", premium)