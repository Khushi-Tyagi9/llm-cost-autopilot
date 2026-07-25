"""
Load test: sends a large volume of prompts through the live API
to generate realistic dashboard data and validate the system under load.

Reuses the labeled dataset (223 prompts), shuffled and repeated to reach
the target count - a standard load-testing approach, not duplicate data
pollution, since each call is independently classified/routed/logged.

Usage: python scripts/load_test.py [target_count]
Default target: 500
Requires the API to be running (Docker container or uvicorn locally).
"""
import sys
import time
import random
import requests
import pandas as pd

API_URL = "http://localhost:8000/v1/completions"
TARGET_COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
DELAY_SECONDS = 0.3  # be gentle on Groq's free tier rate limits


def load_prompts():
    df = pd.read_csv("data/prompts_labeled.csv")
    return df["prompt"].tolist()


def build_test_batch(prompts, target_count):
    """Repeats and shuffles the prompt list to reach target_count."""
    batch = []
    while len(batch) < target_count:
        batch.extend(prompts)
    random.shuffle(batch)
    return batch[:target_count]


def run_load_test():
    prompts = load_prompts()
    batch = build_test_batch(prompts, TARGET_COUNT)

    print(f"Starting load test: {len(batch)} requests to {API_URL}")
    print(f"Estimated time: ~{len(batch) * DELAY_SECONDS / 60:.1f} minutes\n")

    success_count = 0
    error_count = 0
    start_time = time.time()

    for i, prompt in enumerate(batch, 1):
        try:
            response = requests.post(API_URL, json={"prompt": prompt}, timeout=30)
            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1
                print(f"  [{i}] Error {response.status_code}: {response.text[:100]}")
        except requests.exceptions.RequestException as e:
            error_count += 1
            print(f"  [{i}] Request failed: {e}")

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(batch)} ({success_count} ok, {error_count} errors) - {elapsed:.0f}s elapsed")

        time.sleep(DELAY_SECONDS)

    total_time = time.time() - start_time
    print(f"\nDone. {success_count} succeeded, {error_count} failed. Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    run_load_test()