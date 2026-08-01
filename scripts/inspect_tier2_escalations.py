"""
Re-runs the escalated Tier 2 prompts and prints both the cheap and
premium answers side by side, so we can actually read what diverged -
rather than just trusting the escalation count alone.
"""
from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request
from src.verification.judge import verify_response

# the prompts that escalated in the tier 2 quality check
escalated_prompts = [
    "Help me write an email to a hackathon organizer",
    "Spot round statistics for FC/AD/FD (NRI) in top 7 NIFTs",
    "Summarize this product's key features into 3 bullet points: Designed for uncompromising sound and daily versatility, the AuraSound Pro Wireless Headphones combine custom 40mm beryllium drivers for ultra-crisp audio with active hybrid noise cancellation that blocks up to 98% of ambient noise. Built with ultra-lightweight memory foam earcups wrapped in breathable protein leather, they deliver up to 45 hours of continuous playback on a single charge, alongside a rapid-charging feature that gives you 5 hours of listening from just a 10-minute plug-in. Integrated quad-beamforming microphones ensure crystal-clear voice clarity on calls even in windy environments, while Bluetooth 5.3 multipoint connectivity allows seamless switching between your laptop and phone without skipping a beat.",
    "I am changing my internship role to Software Developer. What should I write?",
    "How long does it take for an L1 engineer to reach L6?",
    "Suggest an AI project to impress the SuperKalam internship recruiters",
    "Explain RAG architecture with an example",
]

for i, prompt in enumerate(escalated_prompts, 1):
    print(f"\n{'='*80}")
    print(f"[{i}] PROMPT: {prompt[:100]}")
    print('='*80)

    cheap = send_request(prompt, GROQ_CHEAP)
    print(f"\nCHEAP ANSWER:\n{cheap.text}\n")

    verification = verify_response(prompt, cheap.text, GROQ_PREMIUM)
    print(f"\nPREMIUM ANSWER:\n{verification['premium_text']}\n")

    print(f"JUDGE VERDICT: {verification['judge_verdict']}")