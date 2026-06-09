"""
Idempotent SIP infrastructure provisioning for KoyalAI.

Creates (or finds existing):
  - One SIP inbound trunk per tenant (with phone numbers)
  - One Individual dispatch rule per tenant (one room per call)
  - One SIP outbound trunk per tenant (for campaign dialing)

Idempotent: existing trunks and rules are never duplicated.
Output: .sip_provision.json saved for reference and CI validation.

Usage:
    python scripts/provision_sip.py           # provision all tenants
    python scripts/provision_sip.py --list    # list existing trunks and rules
    python scripts/provision_sip.py --recreate  # delete all + re-provision

Prerequisites:
    1. docker-compose up -d livekit-server livekit-sip redis
    2. LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_WS_URL set in .env
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Centralised tenant SIP configuration
# In production: load from database or tenant config service.
TENANT_SIP_CONFIG: list[dict] = [
    {
        "tenant_id": "tenant_hdfc_bank",
        "inbound_trunk_name": "KoyalAI HDFC Inbound",
        "inbound_numbers": ["+911234567890"],   # Simulated dev number
        "outbound_trunk_name": "KoyalAI HDFC Outbound",
        "outbound_numbers": ["+911234567890"],
        "room_prefix": "tenant_hdfc_bank-inbound-",
        "dispatch_rule_name": "HDFC Individual Dispatch",
    },
    {
        "tenant_id": "tenant_swiggy_support",
        "inbound_trunk_name": "KoyalAI Swiggy Inbound",
        "inbound_numbers": ["+911234567891"],
        "outbound_trunk_name": "KoyalAI Swiggy Outbound",
        "outbound_numbers": ["+911234567891"],
        "room_prefix": "tenant_swiggy_support-inbound-",
        "dispatch_rule_name": "Swiggy Individual Dispatch",
    },
]

OUTPUT_FILE = Path(".sip_provision.json")


async def provision_all(recreate: bool = False) -> dict:
    """Provision SIP trunks and dispatch rules for all tenants.

    Args:
        recreate: If True, deletes existing trunks and dispatch rules
            before reprovisioning.

    Returns:
        Dict mapping tenant_id → provisioning results with trunk IDs.
    """
    from backend.telephony.sip_trunk import SIPTrunkManager  # noqa: PLC0415

    results: dict = {}

    async with SIPTrunkManager() as mgr:
        if recreate:
            logger.info("--recreate: deleting all existing trunks and dispatch rules...")

            # 1. Delete dispatch rules FIRST (they reference trunk IDs)
            rules = await mgr.list_dispatch_rules()
            for rule in rules:
                await mgr.delete_dispatch_rule(rule["id"])
                logger.info("Deleted dispatch rule: %s", rule["id"])

            # 2. Delete inbound trunks
            for trunk in await mgr.list_inbound_trunks():
                await mgr.delete_trunk(trunk.sip_trunk_id)
                logger.info("Deleted inbound trunk: %s (%s)", trunk.sip_trunk_id, trunk.name)

            # 3. Delete outbound trunks
            for trunk in await mgr.list_outbound_trunks():
                await mgr.delete_trunk(trunk.sip_trunk_id)
                logger.info("Deleted outbound trunk: %s (%s)", trunk.sip_trunk_id, trunk.name)

        for cfg in TENANT_SIP_CONFIG:
            tenant_id = cfg["tenant_id"]
            logger.info("Provisioning tenant: %s", tenant_id)

            # Inbound trunk (idempotent)
            inbound = await mgr.create_inbound_trunk_idempotent(
                name=cfg["inbound_trunk_name"],
                numbers=cfg["inbound_numbers"],
                tenant_id=tenant_id,
            )
            logger.info(
                "  Inbound trunk: id=%s created=%s",
                inbound.trunk_id, inbound.created,
            )

            # Dispatch rule (idempotent — one room per call)
            dispatch = await mgr.create_dispatch_rule_idempotent(
                name=cfg["dispatch_rule_name"],
                trunk_id=inbound.trunk_id,
                room_prefix=cfg["room_prefix"],
            )
            logger.info(
                "  Dispatch rule: id=%s created=%s prefix=%s",
                dispatch.rule_id, dispatch.created, dispatch.room_prefix,
            )

            # Outbound trunk (idempotent)
            outbound = await mgr.create_outbound_trunk_idempotent(
                name=cfg["outbound_trunk_name"],
                numbers=cfg["outbound_numbers"],
                tenant_id=tenant_id,
            )
            logger.info(
                "  Outbound trunk: id=%s created=%s",
                outbound.trunk_id, outbound.created,
            )

            results[tenant_id] = {
                "inbound_trunk_id": inbound.trunk_id,
                "inbound_numbers": inbound.numbers,
                "inbound_created": inbound.created,
                "dispatch_rule_id": dispatch.rule_id,
                "room_prefix": dispatch.room_prefix,
                "dispatch_created": dispatch.created,
                "outbound_trunk_id": outbound.trunk_id,
                "outbound_created": outbound.created,
            }

    return results


async def list_existing() -> None:
    """Print all existing SIP trunks and dispatch rules."""
    from backend.telephony.sip_trunk import SIPTrunkManager  # noqa: PLC0415

    async with SIPTrunkManager() as mgr:
        inbound_trunks = await mgr.list_inbound_trunks()
        outbound_trunks = await mgr.list_outbound_trunks()
        rules = await mgr.list_dispatch_rules()

    print(f"\n{'='*55}")
    print(f"Inbound trunks ({len(inbound_trunks)}):")
    for t in inbound_trunks:
        print(f"  {t.sip_trunk_id}: {t.name} — {list(t.numbers)}")
    print(f"\nOutbound trunks ({len(outbound_trunks)}):")
    for t in outbound_trunks:
        print(f"  {t.sip_trunk_id}: {t.name} — {list(t.numbers)}")
    print(f"\nDispatch rules ({len(rules)}):")
    for r in rules:
        print(f"  {r['id']}: trunks={r['trunk_ids']}")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KoyalAI — Idempotent SIP infrastructure provisioning",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing trunks and rules without provisioning",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing trunks and dispatch rules before reprovisioning (destructive!)",
    )
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_existing())
        return

    results = asyncio.run(provision_all(recreate=args.recreate))

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*55}")
    print("✓ SIP provisioning complete (idempotent)")
    print(f"  Tenants provisioned: {len(results)}")
    for tenant_id, info in results.items():
        created_str = "CREATED" if info["inbound_created"] else "existing"
        print(f"\n  {tenant_id}:")
        print(f"    inbound_trunk:  {info['inbound_trunk_id']}  [{created_str}]")
        print(f"    dispatch_rule:  {info['dispatch_rule_id']}")
        print(f"    outbound_trunk: {info['outbound_trunk_id']}")
    print(f"\n  Provisioning config saved to: {OUTPUT_FILE}")
    print(f"{'='*55}")

    # Print the LIVEKIT_SIP_TRUNK_ID to add to .env
    first_outbound = list(results.values())[0]["outbound_trunk_id"]
    print(f"\nAdd to .env:\n  LIVEKIT_SIP_TRUNK_ID={first_outbound}\n")


if __name__ == "__main__":
    sys.exit(main() or 0)