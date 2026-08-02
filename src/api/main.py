"""
FastAPI service exposing the LLM Cost Autopilot router.
Run with: uvicorn src.api.main:app --reload
"""
import numpy as np
import joblib
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from groq import RateLimitError
from pydantic import BaseModel
from src.classifier.features import featurize, is_ungrounded_factual_query
from src.router.config_loader import load_routing_config, resolve_model_for_tier
from src.models.router_client import send_request
from src.models.config import GROQ_PREMIUM
from src.verification.judge import verify_response
from src.verification.scheduler import should_verify
from src.logging.db import init_db, log_request, get_connection

app = FastAPI(title="LLM Cost Autopilot", version="0.1.0")

# Load classifier and config once at startup
model_bundle = joblib.load("src/classifier/model.pkl")
clf = model_bundle["model"]
scaler = model_bundle["scaler"]
needs_scaler = model_bundle["needs_scaler"]

init_db()


class CompletionRequest(BaseModel):
    prompt: str


class CompletionResponse(BaseModel):
    text: str
    tier: int
    model_id: str
    provider: str
    cost: float
    latency: float
    verified: bool
    escalated: bool | None = None


def predict_tier(prompt: str) -> int:
    X = np.array(featurize([prompt]))
    if needs_scaler:
        X = scaler.transform(X)
    return int(clf.predict(X)[0])


def run_verification_and_log(tier: int, response, request_prompt: str, config: dict):
    """
    Runs after the response has already been sent to the user. Verifies
    (if sampled) and logs the request - none of this blocks the actual
    response the user receives.
    """
    verification = None
    sample_rate = config["verification"]["sample_rate"]

    # Only Tier 1 and Tier 2 (non-split) get verification - Tier 3 and
    # Tier 2b already used the premium model directly, so there's no
    # genuine capability gap left to check (see README for reasoning).
    if tier in (1, 2):
        if should_verify(sample_rate):
            threshold = config["verification"]["divergence_threshold"].get(tier, 0.4)
            try:
                verification = verify_response(request_prompt, response.text, GROQ_PREMIUM, threshold)
            except RateLimitError:
                verification = None

    log_request(tier, response, verification, request_prompt)


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    config = load_routing_config()

    tier = predict_tier(request.prompt)

    routing_tier = tier
    if tier == 2 and is_ungrounded_factual_query(request.prompt):
        routing_tier = "2b"

    model_config = resolve_model_for_tier(routing_tier, config)

    try:
        response = send_request(request.prompt, model_config)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Provider rate limit reached. Please retry shortly."
        )

    # Verification and logging happen AFTER this function returns the
    # response to the user - they never wait for it.
    background_tasks.add_task(run_verification_and_log, tier, response, request.prompt, config)

    return CompletionResponse(
        text=response.text,
        tier=tier,
        model_id=response.model_id,
        provider=response.provider,
        cost=response.cost,
        latency=response.latency,
        verified=False,  # not yet known at response time - see /v1/stats for aggregate verification data
        escalated=None,
    )

@app.get("/v1/models")
def list_models():
    config = load_routing_config()
    return config["models"]


@app.get("/v1/stats")
def get_stats():
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_requests,
                SUM(cost) as total_cost,
                AVG(cost) as avg_cost_per_request,
                SUM(escalated) as total_escalated
            FROM requests
        """)
        row = cursor.fetchone()
        return {
            "total_requests": row[0],
            "total_cost": row[1],
            "avg_cost_per_request": row[2],
            "total_escalated": row[3],
        }


@app.put("/v1/routing-config")
def update_routing_config(new_config: dict):
    with open("config/routing.yaml", "w") as f:
        yaml.dump(new_config, f)
    return {"status": "updated"}


@app.get("/")
def root():
    return {"status": "LLM Cost Autopilot is running"}