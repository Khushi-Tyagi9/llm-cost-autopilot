"""
Compares a cheap-tier response against a premium-tier response for the
same prompt, using TF-IDF cosine similarity as a divergence proxy.

No extra LLM calls needed for judging - just one premium generation call,
then a local (free, instant) similarity computation.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.router_client import send_request
from src.models.config import ModelConfig


def compute_similarity(text_a: str, text_b: str) -> float:
    """Returns a 0-1 similarity score. 1.0 = identical content, 0.0 = totally different."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        # happens if both texts are empty/only stopwords
        return 0.0
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


def verify_response(prompt: str, cheap_text: str, premium_config: ModelConfig,
                     divergence_threshold: float = 0.3) -> dict:
    """
    Sends the same prompt to the premium model and compares its answer
    against the cheap model's answer. Returns verification metadata.
    """
    premium_response = send_request(prompt, premium_config)
    similarity = compute_similarity(cheap_text, premium_response.text)
    divergence = 1.0 - similarity
    escalate = divergence > divergence_threshold

    return {
        "premium_text": premium_response.text,
        "premium_cost": premium_response.cost,
        "similarity": similarity,
        "divergence": divergence,
        "escalate": escalate,
    }