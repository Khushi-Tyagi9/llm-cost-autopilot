"""
Runs the same set of prompts through both TF-IDF and LLM-as-judge
verification methods, to compare escalation rates and cost side-by-side.

Resilient to rate limits: saves partial results as it goes, paces
requests to avoid per-minute limits, and stops cleanly on any API error.
"""
import json
import os
import time
import pandas as pd
from groq import RateLimitError, APIStatusError

from src.models.config import GROQ_CHEAP, GROQ_PREMIUM
from src.models.router_client import send_request
from src.verification.judge import compute_similarity
from src.verification.llm_judge import llm_judge_verdict
from src.router.config_loader import load_routing_config

RESULTS_FILE = "data/verification_comparison_results.json"
DELAY_BETWEEN_PROMPTS = 3  # seconds - stays under per-minute token limits


def load_sample_prompts(n=30):
    df = pd.read_csv("data/prompts_labeled.csv")
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return list(zip(sample["prompt"], sample["tier"]))


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
    tfidf_escalations = sum(1 for r in results if r["tfidf_escalate"])
    llm_escalations = sum(1 for r in results if r["llm_escalate"])
    agreements = sum(1 for r in results if r["tfidf_escalate"] == r["llm_escalate"])
    total_judge_cost = sum(r["judge_cost"] for r in results)

    print(f"\n=== RESULTS ({n} prompts completed) ===")
    print(f"TF-IDF escalation rate:     {tfidf_escalations}/{n} ({100*tfidf_escalations/n:.1f}%)")
    print(f"LLM-judge escalation rate:  {llm_escalations}/{n} ({100*llm_escalations/n:.1f}%)")
    print(f"Agreement between methods:  {agreements}/{n} ({100*agreements/n:.1f}%)")
    print(f"Total LLM-judge cost:       ${total_judge_cost:.6f}")


def main():
    config = load_routing_config()
    thresholds = config["verification"]["divergence_threshold"]

    samples = load_sample_prompts(30)
    results = load_existing_results()
    already_done = len(results)

    if already_done > 0:
        print(f"Resuming - {already_done} prompts already completed in a previous run.\n")

    print(f"Target: {len(samples)} prompts total.\n")

    for i, (prompt, tier) in enumerate(samples):
        if i < already_done:
            continue

        try:
            cheap = send_request(prompt, GROQ_CHEAP)
            premium = send_request(prompt, GROQ_PREMIUM)

            similarity = compute_similarity(cheap.text, premium.text)
            divergence = 1.0 - similarity
            tfidf_escalate = divergence > thresholds[int(tier)]

            judge_result = llm_judge_verdict(prompt, cheap.text, premium.text)
            llm_escalate = judge_result["escalate"]

            result = {
                "tier": int(tier),
                "tfidf_escalate": tfidf_escalate,
                "llm_escalate": llm_escalate,
                "judge_cost": judge_result["judge_cost"],
            }
            results.append(result)
            save_results(results)

            agreement_marker = "match" if tfidf_escalate == llm_escalate else "DISAGREE"
            print(f"[{i+1}] Tier {tier} | TF-IDF: {'ESCALATE' if tfidf_escalate else 'ok':9} "
                  f"| LLM-judge: {'ESCALATE' if llm_escalate else 'ok':9} | {agreement_marker}")

            time.sleep(DELAY_BETWEEN_PROMPTS)

        except (RateLimitError, APIStatusError) as e:
            print(f"\nRate/API limit hit at prompt {i+1}/{len(samples)}. Stopping cleanly.")
            print(f"Progress saved to {RESULTS_FILE} - rerun this script later to resume.")
            print_summary(results)
            return
        except Exception as e:
            print(f"\nUnexpected error at prompt {i+1}: {e}")
            print_summary(results)
            return

    print("\nAll prompts completed.")
    print_summary(results)


if __name__ == "__main__":
    main()