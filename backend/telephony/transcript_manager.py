from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder

from backend.telephony.transcript_schema import TranscriptTurn

logger = logging.getLogger(__name__)


class TranscriptManager:
    """
    In-memory transcript subscription registry keyed by LiveKit room name.

    This module is intentionally lightweight:
    - no coupling to SessionManager
    - no dependency on the voice websocket
    - broadcast failures never raise into the audio pipeline
    """

    def __init__(self) -> None:
        # Use a plain dict – sets are created on demand with setdefault
        self._subscriptions: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def register(self, room_name: str, websocket: WebSocket) -> None:
        """
        Register a websocket subscriber for a room.

        The caller is responsible for calling `websocket.accept()`
        before registration. This method only manages the subscription registry.
        """
        async with self._lock:
            subscribers = self._subscriptions.setdefault(room_name, set())
            subscribers.add(websocket)
            total = len(subscribers)

        logger.debug(
            "Transcript subscriber registered: room=%s total=%d",
            room_name,
            total,
        )

    async def unregister(self, room_name: str, websocket: WebSocket) -> None:
        """Remove a websocket subscriber and clean up empty room buckets."""
        async with self._lock:
            subscribers = self._subscriptions.get(room_name)
            if not subscribers:
                return
            subscribers.discard(websocket)
            if not subscribers:
                self._subscriptions.pop(room_name, None)

        logger.debug("Transcript subscriber unregistered: room=%s", room_name)

    async def broadcast(self, room_name: str, payload: TranscriptTurn | dict[str, Any]) -> None:
        """
        Broadcast a transcript event to all current subscribers of a room.

        Payload is encoded defensively so both pydantic models and plain dicts
        are accepted. Delivery is best-effort: a slow or dead websocket never
        blocks the call pipeline.
        """
        # Encode payload once before any locking or sending
        if isinstance(payload, TranscriptTurn):
            event = payload.model_dump(mode="json")
        else:
            event = jsonable_encoder(payload)

        # Snapshot current subscribers under lock
        async with self._lock:
            subscribers = list(self._subscriptions.get(room_name, set()))

        if not subscribers:
            return

        # Send to all subscribers concurrently, isolating failures
        results = await asyncio.gather(
            *(ws.send_json(event) for ws in subscribers),
            return_exceptions=True,
        )

        # Identify dead connections
        stale = [
            ws
            for ws, result in zip(subscribers, results)
            if isinstance(result, Exception)
        ]

        if not stale:
            return

        # Prune dead connections under lock
        async with self._lock:
            current = self._subscriptions.get(room_name)
            if current:
                for ws in stale:
                    current.discard(ws)
                if not current:
                    self._subscriptions.pop(room_name, None)

        # Log each dropped connection (debug level)
        for ws in stale:
            logger.debug(
                "Dropped stale transcript subscriber: room=%s ws_id=%s",
                room_name,
                id(ws),
            )


# ── Singleton 
_broadcaster: TranscriptManager | None = None


def get_transcript_manager() -> TranscriptManager:
    """Return the process-wide TranscriptManager singleton."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = TranscriptManager()
    return _broadcaster