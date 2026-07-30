"""
Runs a larger, varied batch of prompts through the full pipeline
(classify -> route -> verify -> log) to populate the dashboard with
meaningful data instead of just a handful of test requests.
"""
from test_full_pipeline import handle_request

batch_prompts = [
    # Tier 1 - simple
    "What is the capital of Italy?",
    "Convert 10 miles to kilometers.",
    "Extract the phone number from: 'Call 555-0143 for details.'",
    "List the primary colors.",
    "What is 9 multiplied by 7?",

    # Tier 2 - moderate
    "Summarize the differences between REST and GraphQL APIs.",
    "Classify this review as positive or negative: 'Great product, fast shipping, highly recommend.'",
    "Compare the pros and cons of working from home vs. office.",
    "Organize these tasks by priority: fix bug, write docs, deploy update, team meeting.",
    "Summarize the key steps in setting up a CI/CD pipeline.",

    # Tier 3 - complex
    "Design a notification system that avoids spamming users during high-traffic events.",
    "Write a short story about an astronaut stranded alone on Mars.",
    "Propose a strategy to onboard new engineers faster without sacrificing code quality.",
    "Given two conflicting deadlines, decide which project to prioritize and justify it.",
    "Write a persuasive essay on the importance of open-source software.",

    # Mixed - real world style
    "What's the weather API rate limit for the free tier?",
    "Explain the tradeoffs between polling and websockets for real-time updates.",
    "Draft a professional email declining a meeting invite politely.",
    "Analyze why a checkout flow might have a high cart abandonment rate.",
    "Write a product description for a minimalist desk lamp.",
]

for prompt in batch_prompts:
    handle_request(prompt)

print(f"\nDone. Logged {len(batch_prompts)} more requests to data/requests.db")