from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from backend.config import OUTBOUND_MAX_CONCURRENT, load_tenant_config
from backend.cost_tracker import CostTracker
from backend.exceptions import OutboundError, TenantNotFoundError
from backend.voice.stt import SarvamSTT
from backend.voice.tts import SarvamTTS

logger = logging.getLogger(__name__)


class OutboundCaller:
    """Concurrent outbound TTS-synthesise-and-deliver engine.

    Args:
        tenant_id: Validated tenant identifier.

    Example::

        caller = OutboundCaller("tenant_hdfc_bank")
        results = await caller.run_outbound_campaign(
            contact_list=[{"phone": "+91-9800000000", "name": "Priya", "due_amount": "5,000"}],
            script_template="नमस्ते {name} जी, आपकी EMI ₹{due_amount} है।",
            language="hi-IN",
        )
    """

    def __init__(self, tenant_id: str) -> None:
        try:
            self._tenant_cfg = load_tenant_config(tenant_id)
        except TenantNotFoundError as exc:
            raise OutboundError(
                f"Cannot initialise OutboundCaller: {exc}",
                tenant_id=tenant_id,
            ) from exc
        self.tenant_id = tenant_id
        self._tts = SarvamTTS()
        self._cost_tracker = CostTracker()
        logger.info(
            "OutboundCaller initialised for tenant=%s company=%s",
            tenant_id, self._tenant_cfg.get("company_name", "?"),
        )

    async def run_outbound_campaign(
        self,
        contact_list: list[dict[str, Any]],
        script_template: str,
        language: str = "hi-IN",
        max_concurrent: int = OUTBOUND_MAX_CONCURRENT,
    ) -> list[dict]:
        """Run a batched outbound campaign with bounded concurrency.

        Args:
            contact_list: List of contact dicts. Each must have ``"phone"``
                plus any keys referenced by ``script_template``.
            script_template: Python format string. Filled via ``str.format(**contact)``.
            language: BCP-47 language for TTS synthesis.
            max_concurrent: Max simultaneous calls (semaphore bound).

        Returns:
            List of call result dicts — one per contact.
            Failed calls include ``"status": "failed"`` and ``"error"`` keys.

        Raises:
            OutboundError: If contact_list is empty or script_template is blank.
        """
        if not contact_list:
            raise OutboundError("contact_list is empty.", tenant_id=self.tenant_id)
        if not script_template.strip():
            raise OutboundError("script_template is empty.", tenant_id=self.tenant_id)

        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            "Outbound campaign: tenant=%s contacts=%d lang=%s max_concurrent=%d",
            self.tenant_id, len(contact_list), language, max_concurrent,
        )

        async def call_contact(contact: dict) -> dict:
            async with semaphore:
                return await self._make_call(contact, script_template, language)

        tasks = [call_contact(c) for c in contact_list]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[dict] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                phone = contact_list[i].get("phone", "unknown")
                logger.error("Outbound call to %s failed: %s", phone, result)
                results.append({
                    "phone": phone,
                    "status": "failed",
                    "error": str(result),
                })
            else:
                results.append(result)

        completed = sum(1 for r in results if r.get("status") == "completed")
        logger.info(
            "Outbound campaign complete: tenant=%s total=%d completed=%d failed=%d",
            self.tenant_id, len(results), completed, len(results) - completed,
        )
        return results

    async def _make_call(
        self,
        contact: dict[str, Any],
        script_template: str,
        language: str,
    ) -> dict:
        """Process a single outbound call: personalise → TTS → deliver → track cost."""
        session_id = str(uuid.uuid4())
        phone = contact.get("phone", "unknown")
        start_time = time.monotonic()

        # Personalise script
        try:
            personalised_script = script_template.format(**contact)
        except KeyError as exc:
            raise OutboundError(
                f"Script template missing key: {exc}",
                phone=phone,
                template=script_template[:100],
            ) from exc

        # TTS synthesis — native async, no executor needed
        try:
            audio_bytes = await self._tts.asynthesize(personalised_script, language)
        except Exception as exc:
            raise OutboundError(
                f"TTS synthesis failed: {exc}",
                phone=phone,
            ) from exc

        if not audio_bytes:
            raise OutboundError(
                "TTS returned empty audio — call aborted.",
                phone=phone,
            )

        delivery_result = await self._deliver_call(
            session_id=session_id,
            phone=phone,
            audio_bytes=audio_bytes,
            language=language,
        )

        # Track costs
        tts_chars = len(personalised_script)
        await self._cost_tracker.track_tts(self.tenant_id, tts_chars)
        await self._cost_tracker.track_call(self.tenant_id, outcome="completed")

        duration_ms = (time.monotonic() - start_time) * 1000
        cost_inr = round(tts_chars * 0.0015, 4)

        logger.info(
            "Outbound call complete: session=%s phone=%s duration=%.0fms cost_inr=%.4f",
            session_id, phone, duration_ms, cost_inr,
        )

        return {
            "session_id": session_id,
            "phone": phone,
            "status": "completed",
            "language": language,
            "script_chars": tts_chars,
            "audio_size_bytes": len(audio_bytes),
            "cost_inr": cost_inr,
            "duration_ms": round(duration_ms, 1),
            **delivery_result,
        }

    async def _deliver_call(
        self,
        session_id: str,
        phone: str,
        audio_bytes: bytes,
        language: str,
    ) -> dict:
        """Deliver audio to the phone number.

        Phase 3: Stub — logs intent, simulates 50ms network round-trip.
        Phase 6: Replace with LiveKit SIP dial using LiveKit Python SDK.
        """
        logger.info(
            "[STUB] _deliver_call: session=%s phone=%s audio=%d bytes lang=%s "
            "— Phase 6 will wire LiveKit SIP dial here.",
            session_id, phone, len(audio_bytes), language,
        )
        await asyncio.sleep(0.05)  # Simulate network latency
        return {
            "delivery_method": "stub",
            "delivery_status": "delivered",
            "livekit_room": None,    
        }