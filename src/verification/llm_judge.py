from src.models.router_client import send_request
from src.models.config import GROQ_PREMIUM


JUDGE_PROMPT_TEMPLATE = """You are comparing two AI-generated answers to the same question, to check if they are substantively equivalent (same correct information/quality), even if worded differently.

Question: {prompt}

Answer A: {answer_a}

Answer B: {answer_b}

Are these two answers substantively equivalent (same core information, similar quality/correctness), even if phrased differently? Respond with EXACTLY one word: MATCH or DIVERGE.
"""


def llm_judge_verdict(prompt: str, answer_a: str, answer_b: str) -> dict:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, answer_a=answer_a, answer_b=answer_b
    )
    response = send_request(judge_prompt, GROQ_PREMIUM)
    verdict_text = response.text.strip().upper()
    escalate = "DIVERGE" in verdict_text

    return {
        "verdict": verdict_text,
        "escalate": escalate,
        "judge_cost": response.cost,
    }