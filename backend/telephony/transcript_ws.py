"""
FastAPI WebSocket endpoint for live transcript streaming.

URL format: /ws/transcript/{room_name}
Example:    /ws/transcript/tenant_hdfc_bank-inbound-a1b2c3d4

Protocol:
  • Client connects → automatically subscribed to all future transcript events for that room.
  • Server sends transcript events as JSON as they arrive from the LiveKit pipeline.
  • Client may send optional text frames:
      - "ping" → server replies "pong" (helps keep connection alive through proxies/load balancers)
      - any other text is ignored (the stream is read‑only).
  • Client disconnects → automatically unsubscribed.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.telephony.transcript_manager import get_transcript_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcript"])


@router.websocket("/ws/transcript/{room_name}")
async def transcript_ws(websocket: WebSocket, room_name: str) -> None:
    """
    Read-only live transcript channel for a LiveKit room.

    The WebSocket is automatically accepted inside the TranscriptManager
    during registration. The connection stays open, and the server pushes
    transcript events as they become available.

    Args:
        websocket: The FastAPI WebSocket connection.
        room_name: The full LiveKit room name (e.g., "tenant_hdfc_bank-inbound-abc123").
    """
    await websocket.accept()   # explicit handshake

    manager = get_transcript_manager()
    await manager.register(room_name, websocket)   

    try:
        while True:
            message = await websocket.receive_text()
            # Ping handling: case‑insensitive, tolerant of surrounding whitespace
            if message.strip().lower() == "ping":
                await websocket.send_text("pong")
            # All other messages are ignored – the stream is read‑only
    except WebSocketDisconnect:
        logger.debug("[transcript] Client disconnected from %s", room_name)
    except Exception as exc:
        # Unexpected error – log full traceback for debugging
        logger.exception("[transcript] Unexpected error in %s: %s", room_name, exc)
    finally:
        await manager.unregister(room_name, websocket)