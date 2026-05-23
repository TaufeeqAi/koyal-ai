"""
KoyalAI — Ingest all tenant documents into Qdrant.

Usage:
    python scripts/ingest_all.py

Prerequisites:
    - Qdrant running: docker-compose up -d qdrant
    - .env configured with QDRANT_HOST / QDRANT_PORT (defaults: localhost:6333)
    - data/tenant_hdfc_bank/ and data/tenant_swiggy_support/ populated

This script is idempotent — running it twice drops and recreates collections.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.config import TENANTS
from backend.rag.ingestor import MultilingualIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koyalai.ingest_all")


def main() -> None:
    logger.info("=" * 60)
    logger.info("KoyalAI — Multi-Tenant Ingestion Pipeline")
    logger.info("Tenants: %s", TENANTS)
    logger.info("=" * 60)

    ingestor = MultilingualIngestor()
    results: list[dict[str, Any]] = []
    errors: list[tuple[str, Exception]] = []

    overall_start = time.perf_counter()

    for tenant_id in TENANTS:
        logger.info("\n--- Ingesting: %s ---", tenant_id)
        tenant_start = time.perf_counter()
        try:
            result = ingestor.ingest_tenant(tenant_id)
            elapsed = time.perf_counter() - tenant_start
            result["elapsed_seconds"] = round(elapsed, 2)
            results.append(result)
            logger.info(
                "✓ %s | %d chunks | %d vectors | %.1fs",
                tenant_id,
                result["total_chunks"],
                result["total_vectors"],
                elapsed,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("✗ %s | FAILED: %s", tenant_id, exc, exc_info=True)
            errors.append((tenant_id, exc))

    overall_elapsed = time.perf_counter() - overall_start

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  %-30s  collection=%-40s  vectors=%d  time=%.1fs",
            r["tenant_id"],
            r["collection"],
            r["total_vectors"],
            r["elapsed_seconds"],
        )
        for lang, count in r.get("per_language", {}).items():
            logger.info("    └─ %-10s  %d chunks", lang, count)
    if errors:
        logger.info("\nERRORS:")
        for tenant_id, exc in errors:
            logger.error("  ✗ %s: %s", tenant_id, exc)
    logger.info("-" * 60)
    logger.info("Total time: %.1fs", overall_elapsed)
    logger.info("=" * 60)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()