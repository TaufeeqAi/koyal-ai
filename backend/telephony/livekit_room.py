"""
KoyalRoom: per-call LiveKit room lifecycle manager.

Manages a single LiveKit room representing one voice call session.
Connects the KoyalAI agent participant, attaches the AudioBridge, and
handles clean teardown on disconnect or error.

Token grants (security: minimal privilege):
    room_join:         True  — join the room
    can_publish:       True  — publish TTS audio back to caller
    can_subscribe:     True  — subscribe to caller audio
    can_publish_data:  False — no data channels needed 

Room naming convention:
    {tenant_id}-{call_type}-{session_id}
    e.g. "tenant_hdfc_bank-inbound-abc123"

Usage:
    room = KoyalRoom("tenant_hdfc_bank-inbound-abc123", "tenant_hdfc_bank")
    await room.connect()
    # ... call in progress ...
    await room.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from livekit import api, rtc

from backend.config import (
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_WS_URL,
)
from backend.telephony.audio_bridge import LiveKitAudioBridge

logger = logging.getLogger(__name__)

# In-process room registry — maps room_name → KoyalRoom
# Production move to Redis hash for multi-replica deployments.
# Key: room_name → Value: KoyalRoom instance
active_rooms: dict[str, KoyalRoom] = {}
active_rooms_lock = asyncio.Lock()

class KoyalRoom:
    """Manages a single LiveKit room for one KoyalAI voice call.

    Args:
        room_name: Unique room name (convention: ``{tenant_id}-{type}-{id}``).
        tenant_id: Tenant identifier for RAG and cost tracking.
        request_id: Optional correlation ID for tracing webhook → room lifecycle.

    Example:
        >>> koyal = KoyalRoom("tenant_hdfc_bank-inbound-abc", "tenant_hdfc_bank")
        >>> await koyal.connect()
        >>> # ...call in progress...
        >>> await koyal.disconnect()
    """

    def __init__(
            self,
            room_name: str,
            tenant_id: str,
            request_id: Optional[str] = None,
    ) -> None:
        self.room_name = room_name
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.room: rtc.Room = rtc.Room()
        self.bridge: Optional[LiveKitAudioBridge] = None
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect the KoyalAI agent participant to the room and start the bridge.

        Generates a scoped JWT with minimal privilege (can_publish_data=False)
        and connects to the LiveKit server.

        Raises:
            RuntimeError: If the room connection fails.
        """
        if self._connected:
            logger.warning("KoyalRoom '%s' is already connected.", self.room_name)
            return

        # Agent JWT — minimal privilege (can_publish_data=False per security review)
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(f"koyal-agent-{self.tenant_id}")
            .with_name(f"KoyalAI Agent — {self.tenant_id}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=False,  # Minimal privilege — no data channels needed
                )
            )
            .to_jwt()
        )

        try:
            await self.room.connect(LIVEKIT_WS_URL, token)
            self._connected = True
            logger.info(
                "KoyalRoom connected: room='%s' tenant='%s' participant='%s' request_id='%s'",
                self.room_name, self.tenant_id,
                self.room.local_participant.identity,
                self.request_id or "none",
            )
        except Exception as exc:
            raise RuntimeError(
                f"KoyalRoom failed to connect to room '{self.room_name}': {exc}"
            ) from exc

        # Infer call type from room name convention
        call_type = "outbound" if "-outbound-" in self.room_name else "inbound"

        # Attach and start the audio bridge
        self.bridge = LiveKitAudioBridge(
            room=self.room,
            tenant_id=self.tenant_id,
            call_type=call_type,
        )
        try:
            await self.bridge.start()
        except Exception as exc:
            logger.error(
                "KoyalRoom '%s': bridge start failed: %s", self.room_name, exc
            )
            await self.disconnect()
            raise

    async def disconnect(self, outcome: str = "completed") -> None:
        """Disconnect from the LiveKit room and clean up the bridge.

        Args:
            outcome: ``"completed"``, ``"escalated"``, or ``"failed"``.
                Passed to bridge for Prometheus outcome tracking.
        """
        if self.bridge:
            try:
                await self.bridge.stop(outcome=outcome)
            except Exception as exc:
                logger.warning(
                    "KoyalRoom '%s': bridge stop error: %s", self.room_name, exc
                )
            self.bridge = None

        if self._connected:
            try:
                await self.room.disconnect()
                self._connected = False
                logger.info(
                    "KoyalRoom disconnected: room='%s' outcome='%s' request_id=%s",
                    self.room_name, outcome,
                    self.request_id or "none",
                )
            except Exception as exc:
                logger.warning(
                    "KoyalRoom '%s': disconnect error: %s", self.room_name, exc
                )

    @property
    def is_connected(self) -> bool:
        """True if the room is currently connected."""
        return self._connected