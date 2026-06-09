from __future__ import annotations

import logging
import os
import uuid
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv()
logger = logging.getLogger(__name__)

if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WS_URL"):
    os.environ["LIVEKIT_URL"] = os.environ["LIVEKIT_WS_URL"]


async def entrypoint(ctx: JobContext) -> None:
    """Called by livekit-agents for each new room/job.

    Connects to the assigned room as the KoyalAI agent participant,
    launches the LiveKitAudioBridge, and waits until the call ends.

    Args:
        ctx: JobContext provided by livekit-agents framework.
    """
    await ctx.connect()

    room_name: str = ctx.room.name
    logger.info(
        "Worker: new job for room='%s' agent_identity='%s'",
        room_name,
        ctx.room.local_participant.identity if ctx.room.local_participant else "?",
    )

    from backend.telephony.inbound_handler import _extract_tenant_from_room_name  # noqa: PLC0415
    from backend.telephony.livekit_room import KoyalRoom, active_rooms, active_rooms_lock
    from backend.telephony.audio_bridge import LiveKitAudioBridge  # noqa: PLC0415

    try:
        tenant_id = _extract_tenant_from_room_name(room_name)
    except ValueError as exc:
        logger.error( "Cannot determine tenant for room '%s' — aborting job: %s",room_name, exc)
        return
    
    call_type = "outbound" if "-outbound-" in room_name else "inbound"

    # ── Register in shared registry
    request_id = f"worker-{uuid.uuid4().hex[:8]}"
    koyal_room = KoyalRoom(room_name=room_name, tenant_id=tenant_id, request_id=request_id)
    koyal_room.room = ctx.room
    koyal_room._connected = True

    async with active_rooms_lock:
        active_rooms[room_name] = koyal_room

    bridge = LiveKitAudioBridge(
        room=ctx.room,
        tenant_id=tenant_id,
        call_type=call_type,
    )

    koyal_room.bridge = bridge 

    try:
        await bridge.start()
        logger.info(
            "Worker: bridge started for room='%s' tenant='%s'", room_name, tenant_id
        )
        await bridge._stop_event.wait()
    except Exception as exc:
        logger.exception(
            "Worker: bridge error for room='%s': %s", room_name, exc
        )
    finally:
        await bridge.stop(outcome="completed")
        async with active_rooms_lock:
            active_rooms.pop(room_name, None)
        logger.info("Worker: job complete for room='%s'", room_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from backend.config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET  # noqa: PLC0415

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )