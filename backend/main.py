"""
Endpoints
─────────
  WebSocket  /ws/{tenant_id}/{session_id}     — full-duplex voice call
  POST       /api/outbound/campaign           — outbound call campaign
  GET        /api/costs/{tenant_id}           — ₹ cost breakdown (sync Redis read)
  GET        /api/sessions                    — active sessions
  GET        /api/sessions/{session_id}       — single session state
  GET        /health                          — liveness + Redis check
  GET        /metrics                         — Prometheus stub (Phase 4 replaces)

Startup lifecycle
─────────────────
  1. validate_runtime_config() — raises on missing GROQ_API_KEY
  2. Redis ping — warns if unreachable (cost tracking degraded, not fatal)
  3. Log configured tenants
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from backend.config import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    TENANTS,
    validate_runtime_config,
)
from backend.cost_tracker import CostTracker
from backend.exceptions import ConfigValidationError, OutboundError, TenantNotFoundError
from backend.voice.outbound_caller import OutboundCaller
from backend.voice.session_manager import get_session_manager
from backend.voice.websocket_handler import WebSocketVoiceHandler

logger = logging.getLogger(__name__)

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
    version: str = "3.1.0"
    services: dict[str, str]


# ── Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("KoyalAI Phase 3 (v1.1) starting up...")

    # 1. Runtime config validation 
    try:
        validate_runtime_config()
        logger.info("Runtime config: OK")
    except ConfigValidationError as exc:
        logger.error("STARTUP FAILURE — config invalid: %s", exc)
        raise

    # 2. Redis connectivity check
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
    logger.info("Startup complete.")

    yield

    # Shutdown: close shared cost tracker connections
    logger.info("KoyalAI shutting down...")
    try:
        tracker = CostTracker()
        await tracker.close()
    except Exception:
        pass
    logger.info("Shutdown complete.")


# ── App 

app = FastAPI(
    title="KoyalAI Voice API",
    description="Multilingual AI voice agent — Hindi · English · Hinglish · 9 Indian languages",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_methods=["*"],
    allow_headers=["*"],
)

_cost_tracker = CostTracker()   # Module-level shared connection pool


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
        caller = OutboundCaller(request.tenant_id)
        contact_dicts = [c.model_dump() for c in request.contact_list]
        results = await caller.run_outbound_campaign(
            contact_list=contact_dicts,
            script_template=request.script_template,
            language=request.language,
            max_concurrent=request.max_concurrent,
        )
        completed = sum(1 for r in results if r.get("status") == "completed")
        return {
            "campaign_id": str(uuid.uuid4()),
            "tenant_id": request.tenant_id,
            "total": len(results),
            "completed": completed,
            "failed": len(results) - completed,
            "results": results,
        }
    except OutboundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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

    sm = get_session_manager()
    active = sm.list_active_sessions()
    services["active_sessions"] = str(len(active))
    overall = "ok" if services.get("redis") == "ok" else "degraded"
    return HealthResponse(status=overall, services=services)


@app.get("/metrics")
async def metrics_stub() -> dict:
    """Phase 3 stub. Phase 4 replaces this with prometheus_client.make_asgi_app()."""
    sm = get_session_manager()
    return {
        "koyalai_active_sessions": len(sm.list_active_sessions()),
        "note": "Full Prometheus metrics in Phase 4 — /metrics returns text/plain.",
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)