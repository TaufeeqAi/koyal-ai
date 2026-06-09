"""
FastAPI router for LiveKit telephony:
  1. Webhook endpoint — receives room lifecycle events from livekit-server
  2. Token endpoint   — issues JWT access tokens for browser/SIP callers
  3. Rooms endpoint   — lists active rooms for ops dashboard
  4. Health endpoint  — liveness probe (always 200 if process is up)
  5. ready endpoint   — readiness probe (probes upstream LiveKit; 503 if down)

Webhook security:
    LiveKit signs webhooks with the API secret using JWT.
    ``api.WebhookReceiver.receive()`` validates the signature before
    processing any event, preventing spoofed room_started events.

Outcome tracking:
    room_finished events derive their Prometheus outcome ("completed" or
    "failed") from the event's reason field. Kicks and errors are no
    longer misreported as "completed".

"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from livekit import api
from pydantic import BaseModel, Field, field_validator

from backend.config import (
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_WS_URL,
    TENANTS,
)
from backend.telephony.livekit_room import KoyalRoom, active_rooms, active_rooms_lock

logger = logging.getLogger(__name__)

# Maximum webhook payload size 
_MAX_WEBHOOK_BODY_SIZE = 64 * 1024  # 64KB

router = APIRouter(prefix="/telephony", tags=["telephony"])

# Cached WebhookReceiver — stateless, safe to reuse across calls
_webhook_receiver: Optional[api.WebhookReceiver] = None

def _generate_request_id() -> str:
    """Generate a short correlation ID for tracing webhook → room lifecycle."""
    return uuid.uuid4().hex[:8]

def _get_webhook_receiver() -> api.WebhookReceiver:
    """Return a cached WebhookReceiver to avoid reconstructing per webhook."""
    global _webhook_receiver
    if _webhook_receiver is None:
        token_verifier = api.TokenVerifier(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        _webhook_receiver = api.WebhookReceiver(token_verifier)
    return _webhook_receiver

# ── Request / Response Models 

_IDENTITY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
_ROOM_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.{}]{1,128}$")

class TokenRequest(BaseModel):
    """Request body for issuing a caller token."""
    tenant_id: str
    room_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=_ROOM_NAME_PATTERN.pattern,
        description="LiveKit room name. Must be 1-128 chars, alphanumeric with _ - . { }",
    )
    caller_identity: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=_IDENTITY_PATTERN.pattern,
        description="Optional caller identity. 1-64 chars, alphanumeric with _ -",
    )

    @field_validator("tenant_id")
    @classmethod
    def tenant_must_be_known(cls, v: str) -> str:
        """Reject unknown tenant IDs to prevent token issuance for phantom tenants."""
        if v not in TENANTS:
            raise ValueError(f"Unknown tenant_id '{v}'. Known: {TENANTS}")
        return v


class TokenResponse(BaseModel):
    """Response for a caller token request."""
    token: str
    ws_url: str
    room_name: str
    tenant_id: str
    identity: str


class WebhookResponse(BaseModel):
    status: str


# ── Endpoints 

@router.post(
    "/livekit/webhook",
    response_model=WebhookResponse,
    summary="LiveKit webhook receiver",
)
async def livekit_webhook(request: Request) -> WebhookResponse:
    """Process a LiveKit server webhook event.

    Validates the JWT Authorization header via ``api.WebhookReceiver``
    before processing any event to prevent spoofed room_started events.

    Supported events:
        - ``room_started``       — connect a KoyalRoom agent
        - ``participant_joined`` — log and verify caller join
        - ``room_finished``      — disconnect and clean up
        - ``participant_left``   — log event

    Returns:
        ``{"status": "ok"}``

    Raises:
        401: If the webhook JWT signature is invalid.
        413: If the body exceeds 64KB.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_WEBHOOK_BODY_SIZE:
        logger.warning("Webhook body too large: %s bytes", content_length)
        raise HTTPException(status_code=413, detail="Payload too large")
    body = await request.body()
    if len(body) > _MAX_WEBHOOK_BODY_SIZE:
        logger.warning("Webhook body too large: %s bytes", len(body))
        raise HTTPException(status_code=413, detail="Payload too large")
    
    auth_header = request.headers.get("Authorization", "")
    request_id = _generate_request_id()

    # Validate JWT signature using WebhookReceiver 
    try:
        receiver = _get_webhook_receiver()
        event = receiver.receive(body.decode("utf-8"), auth_header)
    except Exception as exc:
        logger.warning(
            "LiveKit webhook: invalid signature: %s (auth_header=%r)",
            exc, auth_header[:50],
        )
        raise HTTPException(
            status_code=401, detail="Invalid webhook signature."
        ) from exc

    event_type: str = getattr(event, "event", "") or ""
    logger.info("LiveKit webhook event: %s request_id=%s", event_type, request_id)

    if event_type == "room_started":
        room_info = getattr(event, "room", None)
        if room_info is None:
            logger.error(
                "LiveKit SDK schema drift: event.room is None for event_type=%s",
                event_type,
            )
        room_name: str = getattr(room_info, "name", "") if room_info else ""
        if not room_name:
            logger.warning("room_started event missing room name.")
            return WebhookResponse(status="ok")

        try:
            tenant_id = _extract_tenant_from_room_name(room_name)
        except ValueError as exc:
            logger.error("Webhook room_started: %s", exc)
            return WebhookResponse(status="error")
        async with active_rooms_lock:
            if room_name in active_rooms:
                logger.info(
                    "room_started: worker already registered '%s' (request_id=%s) — no-op",
                    room_name, request_id,
                )
                return WebhookResponse(status="ok")

            koyal_room = KoyalRoom(room_name=room_name, tenant_id=tenant_id, request_id=request_id,)
            active_rooms[room_name] = koyal_room
        asyncio.create_task(
            _connect_room_safely(koyal_room, room_name)
        )
        logger.info(
            "Room started: name=%s tenant=%s request_id=%s — connecting KoyalRoom.",
            room_name, tenant_id, request_id,
        )

    elif event_type == "room_finished":
        room_info = getattr(event, "room", None)
        if room_info is None:
            logger.error(
                "LiveKit SDK schema drift: event.room is None for event_type=%s",
                event_type,
            )
        room_name = getattr(room_info, "name", "") if room_info else ""
        outcome = _extract_outcome_from_room_finished(event)
        async with active_rooms_lock:
            if room_name in active_rooms:
                koyal_room = active_rooms.pop(room_name)
                asyncio.create_task(
                    koyal_room.disconnect(outcome=outcome)
                )
                logger.info(
                    "Room finished: name=%s outcome=%s request_id=%s — disconnecting KoyalRoom.",
                    room_name, outcome, request_id,
                )

    elif event_type == "participant_joined":
        room_info = getattr(event, "room", None)
        if room_info is None:
            logger.error(
                "LiveKit SDK schema drift: event.room is None for event_type=%s",
                event_type,
            )
        room_name = getattr(room_info, "name", "") if room_info else ""
        participant = getattr(event, "participant", None)
        if participant is None:
            logger.error(
                "LiveKit SDK schema drift: event.participant is None for event_type=%s",
                event_type,
            )
        identity = getattr(participant, "identity", "unknown") if participant else "unknown"
        logger.info(
            "Participant joined: room=%s identity=%s request_id=%s",
            room_name, identity, request_id,
        )
        # verify this is the expected caller, not an unauthorized joiner
        # Production: compare against token-issued identity or room metadata
        if identity.startswith("caller-") or identity.startswith("sip-caller-"):
            logger.info("Verified caller joined room=%s", room_name)
        elif not identity.startswith("koyal-agent-"):
            logger.warning(
                "Unexpected participant joined room=%s identity=%s — possible unauthorized access",
                room_name, identity,
            )

    elif event_type == "participant_left":
        room_info = getattr(event, "room", None)
        if room_info is None:
            logger.error(
                "LiveKit SDK schema drift: event.room is None for event_type=%s",
                event_type,
            )
        room_name = getattr(room_info, "name", "") if room_info else ""
        participant = getattr(event, "participant", None)
        if participant is None:
            logger.error(
                "LiveKit SDK schema drift: event.participant is None for event_type=%s",
                event_type,
            )
        identity = getattr(participant, "identity", "unknown") if participant else "unknown"
        logger.info("Participant left: room=%s identity=%s request_id=%s", room_name, identity, request_id)

        # If the agent is the only participant left, disconnect to free resources
        is_caller = identity.startswith("caller-") or identity.startswith("sip-caller-")
        if is_caller and room_name:
            async with active_rooms_lock:
                if room_name in active_rooms:
                    logger.info("Caller left room=%s — disconnecting KoyalRoom.", room_name)
                    koyal_room = active_rooms.pop(room_name)
                    asyncio.create_task(
                        koyal_room.disconnect(outcome="caller_left")
                    )

    return WebhookResponse(status="ok")


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue a caller JWT token",
)
async def get_caller_token(req: TokenRequest) -> TokenResponse:
    """Issue a LiveKit JWT for a caller participant.

    The token allows the caller to join the specified room, publish their
    microphone audio, and subscribe to the agent's response audio.

    Args:
        req: Token request body with ``tenant_id``, ``room_name``,
             and optional ``caller_identity``.

    Returns:
        ``TokenResponse`` containing the JWT and the LiveKit WebSocket URL.

    Raises:
        422: If ``tenant_id`` is not in ``TENANTS``.
    """
    identity = req.caller_identity or f"caller-{req.tenant_id}-{uuid.uuid4().hex[:8]}"
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(f"Caller — {req.tenant_id}")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=req.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=False,
            )
        )
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )
    logger.info(
        "Issued caller token: identity=%s room=%s tenant=%s",
        identity, req.room_name, req.tenant_id,
    )
    return TokenResponse(
        token=token,
        ws_url=LIVEKIT_WS_URL,
        room_name=req.room_name,
        tenant_id=req.tenant_id,
        identity=identity,
    )


@router.get("/rooms", summary="List active KoyalAI call rooms")
async def list_active_rooms() -> dict:
    """Return all rooms currently tracked by this process.
    
    Includes both connected and connecting rooms for ops visibility.
    """
    async with active_rooms_lock:
        rooms = [
            {
                "room_name": name,
                "tenant_id": room.tenant_id,
                "connected": room.is_connected,
                "connecting": not room.is_connected,
            }
            for name, room in active_rooms.items()
        ]
        connected_count = sum(1 for r in rooms if r["connected"])
    
    return {
        "count": len(rooms),
        "connected_count": connected_count,
        "connecting_count": len(rooms) - connected_count,
        "rooms": rooms,
    }

@router.get("/health", summary="Livekit health check")
async def telephony_health() -> dict[str, Any]:
    """Livekit liveness probe — returns immediately. Pod is alive if this responds.

    Returns active room count and LiveKit URL for ops monitoring dashboards.
   

    Returns:
        JSON with status, active_rooms count, and livekit_url.
    """
    async with active_rooms_lock:
        active = len(active_rooms)
    return {
        "status": "ok",
        "active_rooms": active,
        "livekit_url": LIVEKIT_WS_URL,
    }


@router.get("/ready", summary="LiveKit readiness probe")
async def telephony_ready() -> dict[str, Any]:
    """Readiness probe — actually checks LiveKit server connectivity.
    
    Calls LiveKit API to list rooms. If the server is unreachable or
    returns an error, returns 503 so k8s stops routing traffic.
    """
    try:
        http_url = LIVEKIT_WS_URL.replace("ws://", "http://").replace("wss://", "https://")
        lk_api = api.LiveKitAPI(http_url, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)

        try:
            response = await lk_api.list_rooms(api.ListRoomsRequest())
        finally:
            # close to avoid connection leak
            await lk_api.aclose()

        async with active_rooms_lock:
            active = len(active_rooms)
        return {
            "status": "ok",
            "active_rooms": active,
            "livekit_rooms": len(response.rooms),
            "livekit_url": LIVEKIT_WS_URL,
        }
    except Exception as exc:
        logger.warning("Readiness probe failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "reason": str(exc)},
        )

# ── Private helpers 

async def _connect_room_safely(koyal_room: KoyalRoom, room_name: str) -> None:
    """Connect a KoyalRoom with error logging and cleanup on failure."""
    try:
        await koyal_room.connect()
        logger.info(
            "KoyalRoom '%s' connected (request_id=%s)",
            room_name, koyal_room.request_id or "none",
        )
    except Exception as exc:
        logger.error("KoyalRoom '%s' connect FAILED (request_id=%s): %s", room_name,  koyal_room.request_id or "none", exc, exc_info=True,)
        async with active_rooms_lock:
            active_rooms.pop(room_name, None)

def _extract_tenant_from_room_name(room_name: str) -> str:
    """Extract tenant_id from room name by prefix matching against TENANTS.
    
    Raises ValueError if no known tenant prefix matches — no silent fallback
    to prevent cross-tenant data leakage and billing misattribution.
    
    Convention: ``{tenant_id}-{call_type}-{session_id}``
    
    Args:
        room_name: LiveKit room name.
    
    Returns:
        Tenant ID string.
    
    Raises:
        ValueError: If room name does not start with any known tenant prefix.
    
    Example:
        >>> _extract_tenant_from_room_name("tenant_hdfc_bank-inbound-abc")
        'tenant_hdfc_bank'
    """
    for tenant in TENANTS:
        if room_name.startswith(tenant):
            return tenant
    raise ValueError(
        f"Cannot determine tenant from room name '{room_name}'. "
        f"Known tenants: {TENANTS}"
    )

def _extract_outcome_from_room_finished(event: Any) -> str:
    """Derive a Prometheus outcome from a room_finished event.

    LiveKit's room_finished event may carry a ``reason`` field that hints
    at why the room ended. Kicks, rejections, and errors are reported as
    ``"failed"``; everything else falls through to ``"completed"``.

    Args:
        event: Decoded ``WebhookEvent`` from ``api.WebhookReceiver``.

    Returns:
        One of ``"completed"`` (default) or ``"failed"`` (on error/kick/reject).
    """
    reason = getattr(event, "reason", None) or ""
    if isinstance(reason, str) and reason.lower() in {"error", "failed", "kicked", "rejected"}:
        return "failed"
    return "completed"