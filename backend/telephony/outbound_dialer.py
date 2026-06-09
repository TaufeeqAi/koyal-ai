from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from string import Template
from typing import Optional

from fastapi import APIRouter, HTTPException
from livekit import api
from pydantic import BaseModel, Field, field_validator

from backend.config import (
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_SIP_TRUNK_ID,
    LIVEKIT_WS_URL,
    PRE_SYNTHESIZE_GREETING,
)
from backend.cost_tracker import CostTracker
from backend.exceptions import OutboundError, TTSError
from backend.telephony.livekit_room import KoyalRoom, active_rooms, active_rooms_lock
from backend.voice.tts import SarvamTTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telephony/outbound", tags=["outbound"])

# ── Constants 
_SIP_DIAL_TIMEOUT = 30.0
_SIP_HANGUP_TIMEOUT = 10.0

# ── Registries 

_pending_dials: set[str] = set()  
_pending_dials_lock = asyncio.Lock()

# ── Cached readiness probe client
_ready_lkapi: Optional[api.LiveKitAPI] = None
_ready_lkapi_lock = asyncio.Lock()

async def _get_ready_lkapi() -> api.LiveKitAPI:
    """Get a singleton LiveKitAPI client for readiness checks only."""
    global _ready_lkapi
    if _ready_lkapi is None:
        async with _ready_lkapi_lock:
            if _ready_lkapi is None:
                http_url = LIVEKIT_WS_URL.replace("ws://", "http://").replace("wss://", "https://")
                _ready_lkapi = api.LiveKitAPI(
                    url=http_url,
                    api_key=LIVEKIT_API_KEY,
                    api_secret=LIVEKIT_API_SECRET,
                )
    return _ready_lkapi

# ── Validation patterns 
_PHONE_PATTERN = re.compile(r"^\+\d{10,15}$")
_ROOM_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.{}]{1,128}$")


# ── Models 

class DialRequest(BaseModel):
    phone_number: str = Field(pattern=_PHONE_PATTERN.pattern)
    room_name: str = Field(min_length=1, max_length=128, pattern=_ROOM_NAME_PATTERN.pattern)
    tenant_id: str
    script_text: str = Field(min_length=1, max_length=1000)
    language: str = Field(default="hi-IN", pattern=r"^[a-z]{2}-[A-Z]{2}$")

    @field_validator("tenant_id")
    @classmethod
    def tenant_must_be_known(cls, v: str) -> str:
        from backend.config import TENANTS
        if v not in TENANTS:
            raise ValueError(f"Unknown tenant_id '{v}'")
        return v


@dataclass
class DialResult:
    session_id: str
    phone: str
    room_name: str
    tenant_id: str
    language: str
    status: str = "dialing"
    sip_call_id: Optional[str] = None
    participant_identity: Optional[str] = None
    setup_duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CampaignResult:
    total: int
    dialing: int
    failed: int
    skipped: int
    results: list[DialResult] = field(default_factory=list)


# ── LiveKitSIPOutbound 

class LiveKitSIPOutbound:
    def __init__(
        self,
        livekit_url: Optional[str] = None,
        sip_trunk_id: Optional[str] = None,
    ) -> None:
        http_url = (
            (livekit_url or LIVEKIT_WS_URL)
            .replace("ws://", "http://")
            .replace("wss://", "https://")
        )
        self._lkapi = api.LiveKitAPI(
            url=http_url,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        self._sip_trunk_id = sip_trunk_id or LIVEKIT_SIP_TRUNK_ID
        self._api_lock = asyncio.Lock()
        self._tts = SarvamTTS()
        self._cost_tracker = CostTracker()
        logger.info("LiveKitSIPOutbound initialised (url=%s trunk=%s)", http_url, self._sip_trunk_id)

    async def dial(self, req: DialRequest) -> DialResult:
        if not self._sip_trunk_id:
            raise OutboundError("LIVEKIT_SIP_TRUNK_ID not configured", phone=req.phone_number, tenant_id=req.tenant_id)

        start_time = time.monotonic()
        session_id = str(uuid.uuid4())
        participant_identity = f"sip-caller-{session_id[:8]}"
        sip_call_id: Optional[str] = None
        koyal_room: Optional[KoyalRoom] = None
        sip_created = False

        # Race-proof duplicate detection with _pending_dials
        async with _pending_dials_lock:
            if req.room_name in _pending_dials or req.room_name in active_rooms:
                logger.warning("Duplicate dial blocked: room=%s", req.room_name)
                return DialResult(
                    session_id=session_id, phone=req.phone_number, room_name=req.room_name,
                    tenant_id=req.tenant_id, language=req.language,
                    status="skipped", error="Room already active (duplicate dial)",
                )
            _pending_dials.add(req.room_name)

        try:
            logger.info("Outbound dial: phone=%s room=%s tenant=%s", req.phone_number, req.room_name, req.tenant_id)

            # Optional pre-synthesis (configurable cost vs safety)
            if PRE_SYNTHESIZE_GREETING:
                try:
                    greeting_wav = await asyncio.to_thread(
                        self._tts.synthesize, req.script_text, language_code=req.language,
                    )
                    if not greeting_wav:
                        raise OutboundError("TTS returned empty audio", phone=req.phone_number, tenant_id=req.tenant_id)
                except TTSError as exc:
                    raise OutboundError(f"TTS pre-synthesis failed: {exc}", phone=req.phone_number, tenant_id=req.tenant_id) from exc

            # Create SIP participant 
            try:
                async with asyncio.timeout(_SIP_DIAL_TIMEOUT):
                    async with self._api_lock:
                        sip_participant = await self._lkapi.sip.create_sip_participant(
                            api.CreateSIPParticipantRequest(
                                sip_trunk_id=self._sip_trunk_id,
                                sip_call_to=req.phone_number,
                                room_name=req.room_name,
                                participant_identity=participant_identity,
                                participant_name=f"Caller {req.phone_number}",
                                play_dialtone=True,
                            )
                        )
                sip_call_id = getattr(sip_participant, "sip_call_id", session_id)
                sip_created = True
                logger.info("SIP created: phone=%s room=%s sip_call_id=%s", req.phone_number, req.room_name, sip_call_id)
            except asyncio.TimeoutError:
                raise OutboundError(f"SIP dial timed out after {_SIP_DIAL_TIMEOUT}s", phone=req.phone_number, tenant_id=req.tenant_id)
            except Exception as exc:
                raise OutboundError(f"LiveKit SIP dial failed: {exc}", phone=req.phone_number, tenant_id=req.tenant_id) from exc

            # Register KoyalRoom
            koyal_room = KoyalRoom(room_name=req.room_name, tenant_id=req.tenant_id, request_id=session_id)
            async with active_rooms_lock:
                active_rooms[req.room_name] = koyal_room

            asyncio.create_task(_connect_and_monitor(
                koyal_room=koyal_room, room_name=req.room_name,
                dialer=self, participant_identity=participant_identity,
            ))

            # Non-blocking cost tracking
            try:
                await self._cost_tracker.track_tts(req.tenant_id, len(req.script_text))
            except Exception as exc:
                logger.warning("Cost tracking failed for %s: %s — dial continues", req.tenant_id, exc)

            return DialResult(
                session_id=session_id, phone=req.phone_number, room_name=req.room_name,
                tenant_id=req.tenant_id, language=req.language, status="dialing",
                sip_call_id=sip_call_id, participant_identity=participant_identity,
                setup_duration_ms=round((time.monotonic() - start_time) * 1000, 1),
            )

        except Exception:
            if sip_created and participant_identity:
                await self._hangup_sip_call(req.room_name, participant_identity)
            if koyal_room:
                async with active_rooms_lock:
                    active_rooms.pop(req.room_name, None)
            raise

        finally:
            async with _pending_dials_lock:
                _pending_dials.discard(req.room_name)

    async def dial_campaign(self, contacts, script_template, language="hi-IN", max_concurrent=5, per_tenant=False):
        template = Template(script_template)

        if per_tenant:
            tenant_semaphores = {}
            for c in contacts:
                tid = c.get("tenant_id", "unknown")
                if tid not in tenant_semaphores:
                    tenant_semaphores[tid] = asyncio.Semaphore(max_concurrent)
            sems = {c.get("tenant_id", "unknown"): tenant_semaphores[c.get("tenant_id", "unknown")] for c in contacts}
        else:
            sem = asyncio.Semaphore(max_concurrent)

        async def dial_one(contact):
            tid = contact.get("tenant_id", "unknown")
            async with (sems[tid] if per_tenant else sem):
                return await self._dial_one_contact(contact, template, language)

        tasks = [asyncio.create_task(dial_one(c)) for c in contacts]
        results = await asyncio.gather(*tasks)  # No return_exceptions=True

        return CampaignResult(
            total=len(contacts),
            dialing=sum(1 for r in results if r.status == "dialing"),
            failed=sum(1 for r in results if r.status == "failed"),
            skipped=sum(1 for r in results if r.status == "skipped"),
            results=results,
        )

    async def _dial_one_contact(self, contact, template, language):
        try:
            script = template.safe_substitute(contact)
            if script == template.template:
                logger.warning("Script template had no matching keys for contact %s", contact.get("phone"))

            req = DialRequest(
                phone_number=contact["phone"],
                room_name=f"{contact.get('tenant_id', 'unknown')}-outbound-{uuid.uuid4().hex[:8]}",
                tenant_id=contact.get("tenant_id", "unknown"),
                script_text=script,
                language=language,
            )
            return await self.dial(req)
        except Exception as exc:
            logger.error("Dial failed for %s: %s", contact.get("phone"), exc)
            return DialResult(
                session_id=str(uuid.uuid4()), phone=contact.get("phone", "unknown"),
                room_name="", tenant_id=contact.get("tenant_id", "unknown"),
                language=language, status="failed", error=str(exc),
            )

    async def _hangup_sip_call(self, room_name, participant_identity):
        if not participant_identity:
            return
        try:
            async with asyncio.timeout(_SIP_HANGUP_TIMEOUT):
                async with self._api_lock:
                    await self._lkapi.sip.delete_sip_participant(
                        api.DeleteSIPParticipantRequest(room_name=room_name, participant_identity=participant_identity)
                    )
            logger.info("SIP hung up: room=%s identity=%s", room_name, participant_identity)
        except asyncio.TimeoutError:
            logger.error("SIP hangup timed out: room=%s", room_name)
        except Exception as exc:
            logger.warning("SIP hangup failed: room=%s error=%s", room_name, exc)

    async def close(self):
        async with active_rooms_lock:
            outbound_rooms = [
                (name, room) for name, room in active_rooms.items()
                if "-outbound-" in name
            ]

        for name, room in outbound_rooms:
            try:
                await room.disconnect(outcome="shutdown")
                async with active_rooms_lock:
                    active_rooms.pop(name, None)
            except Exception as exc:
                logger.warning("Cleanup disconnect failed for %s: %s", name, exc)
        try:
            await self._lkapi.aclose()
        except Exception as exc:
            logger.warning("LiveKitSIPOutbound.close() error: %s", exc)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


async def _connect_and_monitor(koyal_room, room_name, dialer, participant_identity):
    try:
        await koyal_room.connect()
        logger.info("KoyalRoom connected: room=%s request_id=%s", room_name, koyal_room.request_id or "none")
    except Exception as exc:
        logger.error("KoyalRoom connect failed: room=%s error=%s", room_name, exc)
        await dialer._hangup_sip_call(room_name, participant_identity)
        async with active_rooms_lock:
            active_rooms.pop(room_name, None)


@router.get("/health")
async def outbound_health():
    async with active_rooms_lock:
        outbound_count = sum(
            1 for name in active_rooms if "-outbound-" in name
        )
    return {
        "status": "ok",
        "service": "outbound_dialer",
        "active_outbound_rooms": outbound_count,
    }


@router.get("/ready")
async def outbound_ready():
    """Readiness probe using a dedicated LiveKit API client - never blocks dials."""
    try:
        lkapi = await _get_ready_lkapi()
        async with asyncio.timeout(5.0):
            # No lock needed – this client is used only for probes
            await lkapi.list_rooms(api.ListRoomsRequest())
        return {"status": "ok", "service": "outbound_dialer"}
    except Exception as exc:
        logger.warning("Outbound readiness probe failed: %s", exc)
        raise HTTPException(status_code=503, detail={"status": "degraded", "reason": str(exc)})