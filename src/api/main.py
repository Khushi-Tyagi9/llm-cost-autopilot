"""
FastAPI service exposing the LLM Cost Autopilot router.
Run with: uvicorn src.api.main:app --reload
"""
import numpy as np
import joblib
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from groq import RateLimitError
from pydantic import BaseModel
from src.classifier.features import featurize, is_ungrounded_factual_query
from src.router.config_loader import load_routing_config, resolve_model_for_tier
from src.models.router_client import send_request
from src.models.config import GROQ_PREMIUM
from src.verification.judge import verify_response
from src.verification.scheduler import should_verify
from src.logging.db import init_db, log_request, get_connection, get_cached_response, store_cached_response, log_cache_hit, log_error

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


def predict_tier_with_confidence(prompt: str) -> tuple[int, float]:
    X = np.array(featurize([prompt]))
    if needs_scaler:
        X = scaler.transform(X)
    tier = int(clf.predict(X)[0])
    if hasattr(clf, "predict_proba"):
        confidence = float(max(clf.predict_proba(X)[0]))
    else:
        confidence = 1.0  # model doesn't support probability estimates
    return tier, confidence


def run_verification_and_log(tier: int, response, request_prompt: str, config: dict, routing_override: str | None = None, classifier_confidence: float | None = None):
    verification = None
    sample_rate = config["verification"]["sample_rate"]

    if tier in (1, 2) and routing_override is None:
        if should_verify(sample_rate):
            threshold = config["verification"]["divergence_threshold"].get(tier, 0.4)
            try:
                verification = verify_response(request_prompt, response.text, GROQ_PREMIUM, threshold)
            except RateLimitError:
                verification = None

    log_request(tier, response, verification, request_prompt, routing_override=routing_override, classifier_confidence=classifier_confidence)


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    cached = get_cached_response(request.prompt)
    if cached:
        background_tasks.add_task(
        log_cache_hit, request.prompt, cached["tier"], cached["model_id"],
        cached["provider"], cached["original_cost"]
    )
        return CompletionResponse(
            text=cached["text"],
            tier=cached["tier"],
            model_id=cached["model_id"],
            provider=cached["provider"],
            cost=0.0,  # cache hit - no new API call made
            latency=0.001,
            verified=False,
            escalated=None,
        )
    config = load_routing_config()

    tier, classifier_confidence = predict_tier_with_confidence(request.prompt)

    routing_tier = tier
    if tier == 2 and is_ungrounded_factual_query(request.prompt):
        routing_tier = "2b"

    model_config = resolve_model_for_tier(routing_tier, config)

    try:
        response = send_request(request.prompt, model_config)
    except RateLimitError:
        background_tasks.add_task(log_error, request.prompt, "rate_limit", tier)
        raise HTTPException(status_code=429, detail="Provider rate limit reached. Please retry shortly.")
    except Exception as e:
        background_tasks.add_task(log_error, request.prompt, str(e)[:500], tier)
        raise HTTPException(status_code=502, detail="Upstream provider error.")
    store_cached_response(request.prompt, response.text, tier, response.model_id, response.provider, response.cost)
    # Verification and logging happen AFTER this function returns the
    # response to the user - they never wait for it.
    override = routing_tier if routing_tier != tier else None
    background_tasks.add_task(run_verification_and_log, tier, response, request.prompt, config, override, classifier_confidence)

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
        row = conn.execute("""
            SELECT COUNT(*), SUM(cost), AVG(cost), SUM(escalated), AVG(classifier_confidence)
            FROM requests WHERE event_type = 'completion'
        """).fetchone()
        cache_hits = conn.execute("SELECT COUNT(*) FROM requests WHERE event_type = 'cache_hit'").fetchone()[0]
        total_errors = conn.execute("SELECT COUNT(*) FROM requests WHERE event_type = 'error'").fetchone()[0]

        return {
            "total_requests": row[0],
            "total_cost": row[1],
            "avg_cost_per_request": row[2],
            "total_escalated": row[3],
            "avg_classifier_confidence": row[4],
            "cache_hits": cache_hits,
            "errors": total_errors,
        }
        


@app.put("/v1/routing-config")
def update_routing_config(new_config: dict):
    with open("config/routing.yaml", "w") as f:
        yaml.dump(new_config, f)
    return {"status": "updated"}


@app.get("/")
def root():
    return {"status": "LLM Cost Autopilot is running"}

@app.get("/health")
def health():
    """Basic liveness check - is the process running at all."""
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    """Readiness check - is the service able to actually serve traffic
    (classifier loaded, database reachable)."""
    checks = {"classifier_loaded": False, "database_reachable": False}

    try:
        checks["classifier_loaded"] = clf is not None
    except NameError:
        checks["classifier_loaded"] = False

    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        checks["database_reachable"] = True
    except Exception:
        checks["database_reachable"] = False

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return JSONResponse(
        status_code=status_code,
        content={"ready": all_ready, "checks": checks}
    )
