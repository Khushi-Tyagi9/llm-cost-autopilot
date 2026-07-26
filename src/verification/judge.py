"""
Compares a cheap-tier response against a premium-tier response for the
same prompt, using an LLM-as-judge verdict.

Switched from TF-IDF similarity after measuring both methods on 30
prompts: TF-IDF flagged 30% of responses as divergent (mostly phrasing
differences on otherwise-correct answers), while LLM-judge flagged 0%
on the same set, at a cost of ~$0.00005 per comparison. See
scripts/compare_verification_methods.py and the README for details.
"""
from src.models.router_client import send_request
from src.models.config import ModelConfig
from src.verification.llm_judge import llm_judge_verdict


def verify_response(prompt: str, cheap_text: str, premium_config: ModelConfig,
                     divergence_threshold: float = 0.3) -> dict:
    """
    Sends the same prompt to the premium model and uses an LLM judge to
    compare its answer against the cheap model's answer.
    """
    premium_response = send_request(prompt, premium_config)

    judge_result = llm_judge_verdict(prompt, cheap_text, premium_response.text)

    return {
        "premium_text": premium_response.text,
        "premium_cost": premium_response.cost,
        "judge_verdict": judge_result["verdict"],
        "judge_cost": judge_result["judge_cost"],
        "escalate": judge_result["escalate"],
    }