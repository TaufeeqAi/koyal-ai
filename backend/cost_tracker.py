from __future__ import annotations

import logging
from typing import Any, Optional

import redis
import redis.asyncio as aioredis

from backend.config import (
    COST_KEY_TTL_DAYS,
    LLM_COST_PER_TOKEN,
    REDIS_DB,
    REDIS_DECODE_RESPONSES,
    REDIS_HOST,
    REDIS_PORT,
    STT_COST_PER_MINUTE,
    TTS_COST_PER_CHAR,
)
from backend.observability.prometheus_metrics import METRICS

logger = logging.getLogger(__name__)

_cost_tracker: Optional[CostTracker] = None

_COST_KEY_TTL_SECONDS: int = COST_KEY_TTL_DAYS * 86_400


def get_cost_tracker() -> CostTracker:
    """Return the singleton CostTracker instance (create once)."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def _key(tenant_id: str, metric: str) -> str:
    """Build a Redis key with the koyalai namespace prefix.

    Example:
        >>> _key("tenant_hdfc_bank", "stt_inr")
        'koyalai:tenant_hdfc_bank:stt_inr'
    """
    return f"koyalai:{tenant_id}:{metric}"


class CostTracker:
    """Per-tenant voice call cost tracker backed by Redis.

    Instantiate once per application; the connection pools are shared.

    Args:
        redis_host: Redis hostname. Defaults to ``REDIS_HOST`` from config.
        redis_port: Redis port. Defaults to ``REDIS_PORT`` from config.

    Example:
        >>> await tracker.track_stt("tenant_hdfc_bank", seconds=4.2)
        >>> costs = tracker.get_tenant_costs("tenant_hdfc_bank")
        >>> costs["stt_cost_inr"]
        0.035
    """

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
    ) -> None:
        self._host = redis_host
        self._port = redis_port
        # Async client: write operations called from WebSocket event loop
        self._async_redis: aioredis.Redis = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            db=REDIS_DB,
            decode_responses=REDIS_DECODE_RESPONSES,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        # Sync client: read operations called from REST endpoints outside event loop
        self._sync_redis: redis.Redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=REDIS_DB,
            decode_responses=REDIS_DECODE_RESPONSES,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        logger.info("CostTracker initialised (redis=%s:%d)", redis_host, redis_port)

    # ── Async write operations (event loop context) 

    async def track_stt(self, tenant_id: str, seconds: float) -> None:
        """Accumulate STT usage and compute ₹ cost atomically.

        Args:
            tenant_id: Tenant for cost attribution.
            seconds: Duration of audio transcribed (from audio byte length estimate).
        """
        if seconds <= 0:
            return
        cost_inr = (seconds / 60.0) * STT_COST_PER_MINUTE
        METRICS.cost_inr.labels(tenant_id=tenant_id, cost_type="stt").inc(cost_inr)
        try:
            pipe = self._async_redis.pipeline()
            pipe.incrbyfloat(_key(tenant_id, "stt_inr"), cost_inr)
            pipe.incrbyfloat(_key(tenant_id, "stt_seconds"), seconds)
            pipe.expire(_key(tenant_id, "stt_inr"), _COST_KEY_TTL_SECONDS)
            pipe.expire(_key(tenant_id, "stt_seconds"), _COST_KEY_TTL_SECONDS)
            await pipe.execute()
            logger.debug(
                "track_stt tenant=%s seconds=%.2f cost_inr=%.4f",
                tenant_id, seconds, cost_inr,
            )
        except Exception as exc:
            logger.error("CostTracker.track_stt failed (silent): %s", exc)

    async def track_tts(self, tenant_id: str, chars: int) -> None:
        """Accumulate TTS usage and compute ₹ cost atomically.

        Args:
            tenant_id: Tenant identifier.
            chars: Number of characters synthesised.
        """
        if chars <= 0:
            return
        cost_inr = chars * TTS_COST_PER_CHAR
        METRICS.cost_inr.labels(tenant_id=tenant_id, cost_type="tts").inc(cost_inr)
        try:
            pipe = self._async_redis.pipeline()
            pipe.incrbyfloat(_key(tenant_id, "tts_inr"), cost_inr)
            pipe.incrbyfloat(_key(tenant_id, "tts_chars"), chars)
            pipe.expire(_key(tenant_id, "tts_inr"), _COST_KEY_TTL_SECONDS)
            pipe.expire(_key(tenant_id, "tts_chars"), _COST_KEY_TTL_SECONDS)
            await pipe.execute()
            logger.debug(
                "track_tts tenant=%s chars=%d cost_inr=%.4f",
                tenant_id, chars, cost_inr,
            )   
        except Exception as exc:
            logger.error("CostTracker.track_tts failed (silent): %s", exc)

    async def track_llm(self, tenant_id: str, tokens: int) -> None:
        """Record LLM token usage. Cost is ₹0 on Groq free tier."""
        if tokens <= 0:
            return
        cost_inr = tokens * LLM_COST_PER_TOKEN
        try:
            pipe = self._async_redis.pipeline()
            pipe.incrbyfloat(_key(tenant_id, "llm_tokens"), tokens)
            pipe.expire(_key(tenant_id, "llm_tokens"), _COST_KEY_TTL_SECONDS)
            if cost_inr > 0:
                pipe.incrbyfloat(_key(tenant_id, "llm_inr"), cost_inr)
                pipe.expire(_key(tenant_id, "llm_inr"), _COST_KEY_TTL_SECONDS)
            await pipe.execute()
            logger.debug("track_llm tenant=%s tokens=%d", tenant_id, tokens)
        except Exception as exc:
            logger.error("CostTracker.track_llm failed (silent): %s", exc)

    async def track_call(self, tenant_id: str, outcome: str = "completed") -> None:
        """Increment call counter for a specific outcome.

        Args:
            tenant_id: Tenant identifier.
            outcome: One of ``"completed"``, ``"escalated"``, ``"failed"``.
        """
        try:
            key = _key(tenant_id, f"calls_{outcome}")
            pipe = self._async_redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, _COST_KEY_TTL_SECONDS)
            await pipe.execute()
        except Exception as exc:
            logger.error("CostTracker.track_call failed (silent): %s", exc)

    # ── Sync read operations (REST endpoint context) 

    def get_tenant_costs(self, tenant_id: str) -> dict[str, Any]:
        """Return aggregated cost breakdown for a tenant (synchronous).

        Intended for use in FastAPI REST endpoints (outside the event loop's
        critical path). Returns zero values on Redis failure.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dict with ``tenant_id``, ``stt_cost_inr``, ``tts_cost_inr``,
            ``llm_tokens``, ``total_cost_inr``, ``stt_seconds``, ``tts_chars``,
            ``calls_completed``, ``calls_escalated``, ``calls_failed``.
        """
        def _get(metric: str) -> float:
            try:
                val = self._sync_redis.get(_key(tenant_id, metric))
                return float(val) if val else 0.0
            except Exception:
                return 0.0

        stt_inr = _get("stt_inr")
        tts_inr = _get("tts_inr")
        return {
            "tenant_id": tenant_id,
            "stt_cost_inr": round(stt_inr, 4),
            "tts_cost_inr": round(tts_inr, 4),
            "total_cost_inr": round(stt_inr + tts_inr, 4),
            "stt_seconds": round(_get("stt_seconds"), 2),
            "tts_chars": int(_get("tts_chars")),
            "llm_tokens": int(_get("llm_tokens")),
            "calls_completed": int(_get("calls_completed")),
            "calls_escalated": int(_get("calls_escalated")),
            "calls_failed": int(_get("calls_failed")),
            "calls_terminated": int(_get("calls_terminated")),
        }


    def sync_gauges(self, tenant_ids: list[str]) -> None:
        """Sync Redis cost totals to Prometheus gauges for each tenant.
        This method is called by the background task in main.py every 30 seconds.
        It reads current Redis cost values (using the sync Redis client) and pushes
        them into the `koyal_cost_inr_current` Prometheus gauge.

        The gauge labels:
            - tenant_id
            - cost_type (stt, tts, total)

        Args:
            tenant_ids: List of tenant identifiers to sync (e.g., TENANTS from config).
        """
        try:
            from backend.observability.prometheus_metrics import METRICS
        except ImportError:
            logger.warning("sync_gauges: prometheus_metrics not available - skipping")
            return
        
        for tenant_id in tenant_ids:
            try:
                costs = self.get_tenant_costs(tenant_id)
                stt_cost = costs["stt_cost_inr"]
                tts_cost = costs["tts_cost_inr"]
                total_cost = costs["total_cost_inr"]

                METRICS.cost_inr_current.labels(tenant_id=tenant_id, cost_type="stt").set(stt_cost)
                METRICS.cost_inr_current.labels(tenant_id=tenant_id, cost_type="tts").set(tts_cost)
                METRICS.cost_inr_current.labels(tenant_id=tenant_id, cost_type="total").set(total_cost)

                logger.debug("sync_gauges: tenant=%s stt=%.2f tts=%.2f total=%.2f",
                            tenant_id, stt_cost, tts_cost, total_cost)
                
            except Exception as e:
                logger.error("sync_gauges failed for tenant %s: %s", tenant_id, e)

    def reset_tenant_costs(self, tenant_id: str) -> None:
        """Delete all cost keys for a tenant (admin / billing-period reset). IRREVERSIBLE."""
        metrics = [
            "stt_inr", "stt_seconds", "tts_inr", "tts_chars",
            "llm_tokens", "llm_inr", "calls_completed", "calls_escalated", "calls_failed",
        ]
        keys = [_key(tenant_id, m) for m in metrics]
        self._sync_redis.delete(*keys)
        logger.warning("CostTracker: reset all costs for tenant=%s", tenant_id)

    async def close(self) -> None:
        """Close async Redis connection gracefully (call in lifespan shutdown)."""
        await self._async_redis.aclose()