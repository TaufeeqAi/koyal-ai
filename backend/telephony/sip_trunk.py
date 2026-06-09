"""
Idempotent SIP trunk provisioning and management via LiveKit's server API.

Idempotency guarantee:
    - create_inbound_trunk_idempotent(): checks for existing trunk by name
      before creating. Re-running provision_sip.py never creates duplicates.
    - create_dispatch_rule_idempotent(): same pattern for dispatch rules.
    - create_outbound_trunk_idempotent(): same pattern for outbound trunks.

This is critical for CI/CD pipelines and rolling deploys: the provisioning
script is safe to run on every deploy without accumulating zombie trunks in
the LiveKit server.

Dev mode:
    livekit-sip Docker container (local PSTN simulation).
    Numbers: +911234567890 (inbound), +911234567891 (outbound).

Production mode:
    Replace ``allowed_addresses`` with your SIP provider's IP ranges
    and ``address`` with your SIP provider's hostname.
    Compatible providers: Twilio SIP, Telnyx, Bandwidth, Vonage.

SIP trunk types:
    Inbound:  receives calls FROM PSTN, routes into LiveKit rooms.
    Outbound: dials phone numbers FROM LiveKit rooms, connects to PSTN.

Usage:
    async with SIPTrunkManager() as mgr:
        result = await mgr.create_inbound_trunk_idempotent(
            name="HDFC Bank Inbound",
            numbers=["+911234567890"],
            tenant_id="tenant_hdfc_bank",
        )
        print(f"trunk_id={result.trunk_id} created={result.created}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from livekit import api

from backend.config import (
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_WS_URL,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses 

@dataclass
class TrunkProvisionResult:
    """Result of a SIP trunk provisioning operation.

    Attributes:
        trunk_id: LiveKit SIP trunk ID (e.g. ``"ST_XXXX"``).
        trunk_name: Human-readable trunk name.
        numbers: List of phone numbers assigned to the trunk.
        created: True if newly created; False if it already existed.
    """
    trunk_id: str
    trunk_name: str
    numbers: list[str]
    created: bool


@dataclass
class DispatchRuleProvisionResult:
    """Result of a SIP dispatch rule provisioning operation.

    Attributes:
        rule_id: LiveKit SIP dispatch rule ID.
        rule_name: Human-readable rule name.
        room_prefix: Room name prefix used for this rule.
        trunk_ids: SIP trunk IDs this rule applies to.
        created: True if newly created; False if it already existed.
    """
    rule_id: str
    rule_name: str
    room_prefix: str
    trunk_ids: list[str]
    created: bool


# ── SIPTrunkManager 

class SIPTrunkManager:
    """Provisions and queries LiveKit SIP trunks with idempotency guarantees.

    Uses ``api.LiveKitAPI`` (async HTTP) for all server-side SIP operations.
    Use ``async with SIPTrunkManager() as mgr:`` to ensure the underlying
    aiohttp.ClientSession is closed automatically.

    Args:
        livekit_url: LiveKit server HTTP URL.
            Defaults to ``LIVEKIT_WS_URL`` with ``ws://`` replaced by ``http://``.

    Example:
        >>> async with SIPTrunkManager() as mgr:
        ...     result = await mgr.create_inbound_trunk_idempotent(
        ...         name="HDFC Bank Inbound",
        ...         numbers=["+911234567890"],
        ...         tenant_id="tenant_hdfc_bank",
        ...     )
        ...     print(result.trunk_id, result.created)
    """

    def __init__(self, livekit_url: Optional[str] = None) -> None:
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
        logger.debug("SIPTrunkManager initialised (url=%s)", http_url)

    async def __aenter__(self) -> "SIPTrunkManager":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ── Idempotent create operations 

    async def create_inbound_trunk_idempotent(
        self,
        name: str,
        numbers: list[str],
        tenant_id: str,
        allowed_addresses: Optional[list[str]] = None,
    ) -> TrunkProvisionResult:
        """Create an inbound SIP trunk, or return the existing one if found.

        Idempotency: checks existing trunks by name before creating.
        Safe to call on every deploy without creating duplicate trunks.

        In development, ``allowed_addresses`` defaults to ``["0.0.0.0/0"]``
        (accept from anywhere). In production, restrict to your SIP
        provider's IP ranges (e.g. Twilio: 54.172.60.0/30).

        Args:
            name: Human-readable trunk name (e.g. ``"HDFC Bank Inbound"``).
            numbers: List of E.164 phone numbers (e.g. ``["+911234567890"]``).
            tenant_id: Tenant ID for metadata (for lookup/cleanup).
            allowed_addresses: Optional CIDR list to restrict inbound SIP.

        Returns:
            ``TrunkProvisionResult`` with trunk ID and creation status.

        Raises:
            RuntimeError: If LiveKit API call fails.
        """
        existing = await self._find_inbound_trunk_by_name(name)
        if existing is not None:
            logger.info(
                "Inbound trunk already exists: name=%s id=%s", name, existing.sip_trunk_id
            )
            return TrunkProvisionResult(
                trunk_id=existing.sip_trunk_id,
                trunk_name=name,
                numbers=list(existing.numbers),
                created=False,
            )

        try:
            result = await self._lkapi.sip.create_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(
                    trunk=api.SIPInboundTrunkInfo(
                        name=name,
                        numbers=numbers,
                        allowed_addresses=allowed_addresses or ["0.0.0.0/0"],
                        metadata=f"tenant_id={tenant_id}",
                    )
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create SIP inbound trunk '{name}': {exc}"
            ) from exc

        logger.info(
            "Inbound trunk created: name=%s id=%s numbers=%s",
            name, result.sip_trunk_id, numbers,
        )
        return TrunkProvisionResult(
            trunk_id=result.sip_trunk_id,
            trunk_name=name,
            numbers=numbers,
            created=True,
        )

    async def create_dispatch_rule_idempotent(
        self,
        name: str,
        trunk_id: str,
        room_prefix: str,
    ) -> DispatchRuleProvisionResult:
        """Create a SIPDispatchRuleIndividual, or return existing if found.

        ``SIPDispatchRuleIndividual`` creates one room per inbound call,
        named ``{room_prefix}{uuid}``. The webhook handler extracts the
        tenant_id from the room name prefix on ``room_started``.

        Args:
            name: Human-readable rule name (e.g. ``"HDFC Individual Dispatch"``).
            trunk_id: ID of the inbound trunk this rule applies to.
            room_prefix: Room name prefix (e.g. ``"tenant_hdfc_bank-inbound-"``).

        Returns:
            ``DispatchRuleProvisionResult``.

        Raises:
            RuntimeError: If LiveKit API call fails.
        """
        existing = await self._find_dispatch_rule_by_name(name)
        if existing is not None:
            logger.info(
                "Dispatch rule already exists: name=%s id=%s",
                name, existing.sip_dispatch_rule_id,
            )
            return DispatchRuleProvisionResult(
                rule_id=existing.sip_dispatch_rule_id,
                rule_name=name,
                room_prefix=room_prefix,
                trunk_ids=[trunk_id],
                created=False,
            )

        try:
            rule = api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                    room_prefix=room_prefix,
                )
            )
            request = api.CreateSIPDispatchRuleRequest(
                dispatch_rule=api.SIPDispatchRuleInfo(
                    rule=rule,
                    name=name,
                    trunk_ids=[trunk_id],
                    metadata=f"name={name}",
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build SIP dispatch rule request '{name}': {exc}"
            ) from exc

        try:
            result = await self._lkapi.sip.create_dispatch_rule(request)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create SIP dispatch rule '{name}': {exc}"
            ) from exc

        logger.info(
            "Dispatch rule created: name=%s id=%s prefix=%s trunk=%s",
            name, result.sip_dispatch_rule_id, room_prefix, trunk_id,
        )
        return DispatchRuleProvisionResult(
            rule_id=result.sip_dispatch_rule_id,
            rule_name=name,
            room_prefix=room_prefix,
            trunk_ids=[trunk_id],
            created=True,
        )

    async def create_outbound_trunk_idempotent(
        self,
        name: str,
        numbers: list[str],
        tenant_id: str,
        sip_provider_address: str = "livekit-sip:5060",
    ) -> TrunkProvisionResult:
        """Create an outbound SIP trunk for campaign dialing, or return existing.

        Args:
            name: Human-readable trunk name.
            numbers: Caller ID numbers (shown on recipient's phone).
            tenant_id: Tenant identifier for metadata.
            sip_provider_address: SIP server address. Dev: ``"livekit-sip:5060"``.
                Production: ``"sip.twilio.com"`` or your provider's hostname.

        Returns:
            ``TrunkProvisionResult``.
        """
        # Check for existing outbound trunk
        try:
            response = await self._lkapi.sip.list_sip_outbound_trunk(
                api.ListSIPOutboundTrunkRequest()
            )
            for trunk in response.items:
                if trunk.name == name:
                    logger.info(
                        "Outbound trunk already exists: name=%s id=%s",
                        name, trunk.sip_trunk_id,
                    )
                    return TrunkProvisionResult(
                        trunk_id=trunk.sip_trunk_id,
                        trunk_name=name,
                        numbers=list(trunk.numbers),
                        created=False,
                    )
        except Exception as exc:
            logger.warning("Could not list outbound trunks for idempotency check: %s", exc)

        try:
            result = await self._lkapi.sip.create_sip_outbound_trunk(
                api.CreateSIPOutboundTrunkRequest(
                    trunk=api.SIPOutboundTrunkInfo(
                        name=name,
                        address=sip_provider_address,
                        numbers=numbers,
                        metadata=f"tenant_id={tenant_id}",
                    )
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create SIP outbound trunk '{name}': {exc}"
            ) from exc

        logger.info(
            "Outbound trunk created: name=%s id=%s", name, result.sip_trunk_id
        )
        return TrunkProvisionResult(
            trunk_id=result.sip_trunk_id,
            trunk_name=name,
            numbers=numbers,
            created=True,
        )

    # ── Listing + cleanup 

    async def list_inbound_trunks(self) -> list:
        """Return all SIP inbound trunk objects from the server.

        Returns:
            List of ``SIPInboundTrunkInfo`` objects. Returns ``[]`` on error.
        """
        try:
            response = await self._lkapi.sip.list_sip_inbound_trunk(
                api.ListSIPInboundTrunkRequest()
            )
            return response.items
        except Exception as exc:
            logger.error("Failed to list SIP inbound trunks: %s", exc)
            return []

    async def list_outbound_trunks(self) -> list:
        """Return all SIP outbound trunk objects from the server.

        Returns:
            List of ``SIPOutboundTrunkInfo`` objects. Returns ``[]`` on error.
        """
        try:
            response = await self._lkapi.sip.list_sip_outbound_trunk(
                api.ListSIPOutboundTrunkRequest()
            )
            return response.items
        except Exception as exc:
            logger.error("Failed to list SIP outbound trunks: %s", exc)
            return []

    async def list_dispatch_rules(self) -> list[dict]:
        """Return a list of dispatch rule summaries.

        Returns:
            List of dicts with ``id``, ``trunk_ids``, and ``metadata``.
        """
        try:
            resp = await self._lkapi.sip.list_sip_dispatch_rule(
                api.ListSIPDispatchRuleRequest()
            )
            return [
                {
                    "id": r.sip_dispatch_rule_id,
                    "trunk_ids": list(r.trunk_ids),
                    "metadata": r.metadata,
                }
                for r in resp.items
            ]
        except Exception as exc:
            logger.error("Failed to list dispatch rules: %s", exc)
            return []

    async def delete_dispatch_rule(self, rule_id: str) -> bool:
        """Delete a SIP dispatch rule by ID.

        Args:
            rule_id: Dispatch rule ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            await self._lkapi.sip.delete_sip_dispatch_rule(
                api.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rule_id)
            )
            logger.info("Dispatch rule deleted: id=%s", rule_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete dispatch rule '%s': %s", rule_id, exc)
            return False

    async def delete_trunk(self, trunk_id: str) -> bool:
        """Delete any SIP trunk (inbound or outbound) by ID.

        LiveKit uses a unified trunk ID space; the same endpoint deletes
        both inbound and outbound trunks.

        Args:
            trunk_id: Trunk ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            await self._lkapi.sip.delete_sip_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=trunk_id)
            )
            logger.info("SIP trunk deleted: id=%s", trunk_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete SIP trunk '%s': %s", trunk_id, exc)
            return False

    async def close(self) -> None:
        """Close the LiveKit API client and its underlying HTTP session.

        Must be called when the manager is no longer needed to avoid
        aiohttp connection leaks.
        """
        try:
            await self._lkapi.aclose()
            logger.debug("SIPTrunkManager: LiveKitAPI closed.")
        except Exception as exc:
            logger.warning("SIPTrunkManager.close() error: %s", exc)

    # ── Private helpers 

    async def _find_inbound_trunk_by_name(
        self, name: str
    ) -> Optional[api.SIPInboundTrunkInfo]:
        """Return the first inbound trunk with the given name, or None."""
        response = await self._lkapi.sip.list_sip_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        for trunk in response.items:
            if trunk.name == name:
                return trunk
        return None

    async def _find_dispatch_rule_by_name(self, name: str):
        """Return the first dispatch rule whose metadata contains the name, or None."""
        response = await self._lkapi.sip.list_sip_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        for rule in response.items:
            if f"name={name}" in (rule.metadata or ""):
                return rule
        return None