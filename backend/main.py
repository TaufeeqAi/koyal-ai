from __future__ import annotations

import logging
import time
import uuid
import asyncio 
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from backend.config import (
    GUARDRAILS_ENABLED,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    TENANTS,
    validate_runtime_config,
)
from backend.cost_tracker import get_cost_tracker
from backend.exceptions import ConfigValidationError
from backend.rag.embedder import get_embedder
from backend.rag.retriever import get_retriever
from backend.safety.emergency_keywords import get_default_detector
from backend.voice.session_manager import get_session_manager
from backend.voice.websocket_handler import WebSocketVoiceHandler
from backend.observability import (
    flush_langfuse, init_langfuse, get_langfuse_client, is_langfuse_available
)
from backend.observability.prometheus_metrics import PrometheusMiddleware
from prometheus_client import make_asgi_app, REGISTRY
from backend.telephony.inbound_handler import router as telephony_router
from backend.telephony.outbound_dialer import router as outbound_router
from backend.telephony.outbound_dialer import LiveKitSIPOutbound
import backend.groq_patch

logger = logging.getLogger(__name__)

_cost_sync_task: Optional[asyncio.Task] = None

async def _cost_gauge_sync_loop() -> None:
    """Background task: sync Redis cost totals to Prometheus Gauges every 30s.

    Runs indefinitely until the FastAPI process shuts down. Non-fatal errors
    are logged at DEBUG level and retried on the next interval.

    Design note: 30s lag is acceptable for cost dashboards. Real-time cost
    is available via /api/costs/{tenant_id} which reads Redis directly.
    """
    while True:
        await asyncio.sleep(30)
        try:
            _cost_tracker.sync_gauges(TENANTS)
        except Exception as exc:
            logger.debug("Cost gauge sync error (non-fatal): %s", exc)

# ── Request / Response models 

class OutboundContact(BaseModel):
    phone: str
    name: str
    model_config = {"extra": "allow"}   # Allow arbitrary template fields

    @field_validator("phone")
    @classmethod
    def phone_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("phone must not be empty")
        return v.strip()


class OutboundCampaignRequest(BaseModel):
    tenant_id: str
    contact_list: list[OutboundContact]
    script_template: str
    language: str = "hi-IN"
    max_concurrent: int = 5

    @field_validator("tenant_id")
    @classmethod
    def tenant_known(cls, v: str) -> str:
        if v not in TENANTS:
            raise ValueError(f"Unknown tenant_id '{v}'. Known: {TENANTS}")
        return v

    @field_validator("script_template")
    @classmethod
    def script_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("script_template must not be empty")
        return v

    @field_validator("max_concurrent")
    @classmethod
    def concurrent_range(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("max_concurrent must be between 1 and 50")
        return v


class HealthResponse(BaseModel):
    status: str
    version: str = "4.0.0"
    services: dict[str, str]


# ── Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global _cost_sync_task

    from backend.logging_config import setup_logging
    setup_logging()
    
    logger.info("KoyalAI Starting up...")

    # 1. Runtime config validation 
    try:
        validate_runtime_config()
        logger.info("Runtime config: OK")
    except ConfigValidationError as exc:
        logger.error("STARTUP FAILURE — config invalid: %s", exc)
        raise

    # 2. Langfuse client initialisation
    try:
        lf = init_langfuse()
        status = "enabled" if is_langfuse_available() else "disabled (no API keys)"
        logger.info("Langfuse: %s", status)
    except Exception as exc:
        logger.warning("Langfuse init warning (non-fatal): %s", exc)

    # 3. Redis connectivity check
    try:
        r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        await r.ping()
        await r.aclose()
        logger.info("Redis: OK at %s:%d", REDIS_HOST, REDIS_PORT)
    except Exception as exc:
        logger.warning(
            "Redis unreachable at %s:%d: %s — cost tracking degraded.",
            REDIS_HOST, REDIS_PORT, exc,
        )

    logger.info("Configured tenants: %s", TENANTS)

    # 4. Start cost gauge sync background task
    _cost_sync_task = asyncio.create_task(_cost_gauge_sync_loop())
    logger.info("Cost gauge sync task started (interval=30s)")

    # 5. Pre-load heavy ML models to eliminate cold-start latency
    logger.info("=" * 60)
    logger.info("PRE-LOADING ML MODELS (this may take 10-30s)...")
    logger.info("=" * 60)

    preloads = [
        ("LaBSE embedder",    lambda: get_embedder()),
        ("Cross-encoder",     lambda: get_retriever()),
        ("Emergency detector", lambda: get_default_detector()),
    ]
    logger.info("Pre-loading emergency detector...")
    for name, loader in preloads:
        start = time.monotonic()
        try:
            loader()
            elapsed = time.monotonic() - start
            logger.info("✓ %s pre-loaded (%.1fs)", name, elapsed)
        except Exception as exc:
            logger.error("✗ %s pre-load failed: %s", name, exc)
            logger.warning("  First request will pay cold-start penalty")

    logger.info("=" * 60)
    logger.info("ALL MODELS PRE-LOADED — READY FOR CALLS")
    logger.info("=" * 60)

    # 6. Pre-load Guardrails to avoid 60s delay on first request
    if GUARDRAILS_ENABLED:
        logger.info("Pre-loading NeMo Guardrails...")
        try:
            from backend.safety.guardrails_handler import get_guardrails_handler
            get_guardrails_handler()
            logger.info("Guardrails pre-loaded.")
        except Exception as exc:
            logger.warning("Guardrails pre-load failed: %s", exc)
    else:
        logger.info("Guardrails disabled — skipping pre-load.")
    
    yield

    # ── Shutdown
    logger.info("KoyalAI shutting down...")
    if _cost_sync_task and not _cost_sync_task.done():
        _cost_sync_task.cancel()
        try:
            await _cost_sync_task
        except asyncio.CancelledError:
            pass
    try:
        await flush_langfuse()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", exc)

    try:
        await get_cost_tracker().close()
    except Exception:
        pass

    logger.info("Shutdown complete.")


# ── App 

app = FastAPI(
    title="KoyalAI Voice API",
    description="Multilingual AI voice agent — Hindi · English · Hinglish · 9 Indian languages",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3002",
        "http://localhost:8000",
    ],    
    allow_methods=["*"],
    allow_headers=["*"],
)

_cost_tracker = get_cost_tracker() 

app.add_middleware(PrometheusMiddleware)

metrics_app = make_asgi_app(registry=REGISTRY)
app.mount("/metrics/", metrics_app)

app.include_router(telephony_router)
app.include_router(outbound_router)

# ── WebSocket endpoint 

@app.websocket("/ws/{tenant_id}/{session_id}")
async def voice_websocket(websocket: WebSocket, tenant_id: str, session_id: str) -> None:
    if tenant_id not in TENANTS:
        await websocket.accept()
        await websocket.send_text(
            f'{{"type":"error","message":"Unknown tenant: {tenant_id}"}}'
        )
        await websocket.close(code=4004)
        logger.warning("WebSocket rejected: unknown tenant=%s", tenant_id)
        return

    handler = WebSocketVoiceHandler(
        websocket=websocket,
        tenant_id=tenant_id,
        session_id=session_id,
        call_type="inbound",
    )
    await handler.run()


# ── REST endpoints 

@app.post("/api/outbound/campaign", status_code=202)
async def run_outbound_campaign(request: OutboundCampaignRequest) -> dict:
    try:
        async with LiveKitSIPOutbound() as dialer:
            contact_dicts = [
                {**c.model_dump(), "tenant_id": request.tenant_id}
                for c in request.contact_list
            ]
            campaign = await dialer.dial_campaign(
                contacts=contact_dicts,
                script_template=request.script_template,
                language=request.language,
                max_concurrent=request.max_concurrent,
            )

        return {
            "campaign_id": str(uuid.uuid4()),
            "tenant_id": request.tenant_id,
            "total": campaign.total,
            "dialing": campaign.dialing,
            "failed": campaign.failed,
            "skipped": campaign.skipped,
            "results": [
                {
                    "phone": r.phone,
                    "status": r.status,
                    "room_name": r.room_name,
                    "sip_call_id": r.sip_call_id,
                    "error": r.error,
                }
                for r in campaign.results
            ],
        }
    except Exception as exc:
        logger.exception("Outbound campaign error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Campaign failed: {exc}") from exc

@app.get("/api/costs/{tenant_id}")
def get_tenant_costs(tenant_id: str) -> dict:
    """Synchronous endpoint — uses sync Redis read; no event-loop overhead."""
    if tenant_id not in TENANTS:
        raise HTTPException(status_code=404, detail=f"Unknown tenant: {tenant_id}")
    return _cost_tracker.get_tenant_costs(tenant_id)


@app.get("/api/sessions")
async def list_active_sessions(tenant_id: Optional[str] = Query(default=None)) -> dict:
    sm = get_session_manager()
    sessions = sm.list_active_sessions(tenant_id=tenant_id)
    return {"count": len(sessions), "sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    sm = get_session_manager()
    try:
        session = await sm.get_session(session_id)
        return session.to_dict()
    except Exception:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    services: dict[str, str] = {}
    try:
        r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        await r.ping()
        await r.aclose()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"

    lf = get_langfuse_client()
    services["langfuse"] = "enabled" if is_langfuse_available() else "disabled"

    sm = get_session_manager()
    active = sm.list_active_sessions()
    services["active_sessions"] = str(len(active))

    overall = "ok" if services.get("redis") == "ok" else "degraded"
    return HealthResponse(status=overall, services=services)


@app.post("/api/admin/populate-metrics")
async def populate_metrics():
    """Populate observability metrics for Grafana testing."""
    from scripts.populate_observability import main
    result = await main()
    return {
        "status": "ok",
        "populated": result,
        "note": "Metrics now visible at /metrics/ and in Grafana",
    }

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)