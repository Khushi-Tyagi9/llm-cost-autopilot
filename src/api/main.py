"""
FastAPI service exposing the LLM Cost Autopilot router.
Run with: uvicorn src.api.main:app --reload
"""
import numpy as np
import joblib
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.classifier.features import featurize
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


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    config = load_routing_config()
    tier = predict_tier(request.prompt)
    model_config = resolve_model_for_tier(tier, config)

    response = send_request(request.prompt, model_config)

    verification = None
    sample_rate = config["verification"]["sample_rate"]
    if should_verify(sample_rate):
        threshold = config["verification"]["divergence_threshold"][tier]
        verification = verify_response(request.prompt, response.text, GROQ_PREMIUM, threshold)

    log_request(tier, response, verification, request.prompt)

    return CompletionResponse(
        text=response.text,
        tier=tier,
        model_id=response.model_id,
        provider=response.provider,
        cost=response.cost,
        latency=response.latency,
        verified=verification is not None,
        escalated=verification["escalate"] if verification else None,
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