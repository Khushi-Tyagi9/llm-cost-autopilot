"""
Tier-2-specific quality check: takes 30 Tier 2 (moderate) prompts
specifically, and checks whether the cheap model's answers hold up
against the premium model's answers using the LLM-judge.

This exists because the overall 30-prompt comparison mixed all tiers
together - it never specifically confirmed whether routing Tier 2 to
the cheap model (the biggest single driver of the cost savings number)
is actually producing acceptable quality, or just cheap output.
"""
import json
import os
import time
import pandas as pd
from groq import RateLimitError, APIStatusError

from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request
from src.verification.judge import verify_response

RESULTS_FILE = "data/tier2_quality_results.json"
DELAY_BETWEEN_PROMPTS = 3


def load_tier2_prompts(n=30):
    df = pd.read_csv("data/prompts_labeled.csv")
    tier2 = df[df["tier"] == 2]
    sample = tier2.sample(n=min(n, len(tier2)), random_state=42)
    return sample["prompt"].tolist()


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results):
    n = len(results)
    if n == 0:
        print("No results yet.")
        return
    escalations = sum(1 for r in results if r["escalate"])
    print(f"\n=== TIER 2 QUALITY CHECK ({n} prompts) ===")
    print(f"Escalation rate: {escalations}/{n} ({100*escalations/n:.1f}%)")
    print(f"This means {n - escalations}/{n} cheap-tier answers were judged")
    print(f"substantively equivalent to the premium answer on moderate tasks.")


def main():
    prompts = load_tier2_prompts(30)
    results = load_existing_results()
    already_done = len(results)

    if already_done > 0:
        print(f"Resuming - {already_done} prompts already completed.\n")

    print(f"Target: {len(prompts)} Tier 2 prompts.\n")

    for i, prompt in enumerate(prompts):
        if i < already_done:
            continue

        try:
            cheap = send_request(prompt, GROQ_CHEAP)
            verification = verify_response(prompt, cheap.text, GROQ_PREMIUM)

            result = {
                "prompt_preview": prompt[:80],
                "judge_verdict": verification["judge_verdict"],
                "escalate": verification["escalate"],
            }
            results.append(result)
            save_results(results)

            status = "ESCALATE" if verification["escalate"] else "ok"
            print(f"[{i+1}] {status:9} | {prompt[:60]}")

            time.sleep(DELAY_BETWEEN_PROMPTS)

        except (RateLimitError, APIStatusError):
            print(f"\nRate limit hit at prompt {i+1}/{len(prompts)}. Stopping cleanly.")
            print_summary(results)
            return
        except Exception as e:
            print(f"\nUnexpected error at prompt {i+1}: {e}")
            print_summary(results)
            return

    print("\nAll Tier 2 prompts completed.")
    print_summary(results)


if __name__ == "__main__":
    main()