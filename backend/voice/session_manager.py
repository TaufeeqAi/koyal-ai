"""
Per-call session lifecycle management.

Design
──────
* ``CallSession`` — typed dataclass with a 7-state enum (ACTIVE, LISTENING,
  PROCESSING, SPEAKING, ENDED, TIMED_OUT, ESCALATED). Rich to_dict() for API
  responses with native bool/int types (not string-cast).
* ``SessionManager`` — asyncio.Lock protected dict store. Provides:
    - ``create_session()``         — register and return a new session
    - ``get_session()``            — retrieve (raises SessionError if missing)
    - ``update_session(**kwargs)`` — bulk-update fields
    - ``acquire(session_id)``      — context manager for safe concurrent updates
    - ``end_session()``            — mark as ended
    - ``list_active_sessions()``   — for /api/sessions and /health
    - ``load_session_strikes()``   — hydrate harmful_attempt_count from Redis
    - ``save_session_strikes()``   — persist harmful_attempt_count to Redis
* ``get_session_manager()``        — process-level singleton factory

"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.config import (
    MAX_CONCURRENT_SESSIONS,
    REDIS_DB,
    REDIS_DECODE_RESPONSES,
    REDIS_HOST,
    REDIS_PORT,
    SESSION_TIMEOUT_SECONDS,
)
from backend.exceptions import SessionError

logger = logging.getLogger(__name__)

# TTL for session strike keys: session timeout + 5 min buffer for clock skew
_SESSION_STRIKE_TTL_SECONDS: int = SESSION_TIMEOUT_SECONDS + 300


def _session_strike_key(session_id: str) -> str:
    """Build Redis key for session strike count.

    Example:
        >>> _session_strike_key("sess_001")
        'session:sess_001:strikes'
    """
    return f"session:{session_id}:strikes"


class SessionState(Enum):
    """Lifecycle states for a voice call session."""
    ACTIVE = "active"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ENDED = "ended"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"


@dataclass
class CallSession:
    """Mutable call state container for one voice session.

    Args:
        session_id: Unique session identifier (UUID string).
        tenant_id: Tenant this call belongs to.
        call_type: ``"inbound"`` or ``"outbound"``.
    """
    session_id: str
    tenant_id: str
    call_type: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Mutable lifecycle fields
    state: SessionState = SessionState.ACTIVE
    language: Optional[str] = None
    is_code_mixed: bool = False
    end_time: Optional[datetime] = None
    outcome: Optional[str] = None      # "completed" | "escalated" | "failed" | "timed_out" | "terminated"
    last_error: Optional[str] = None

    # Running usage counters
    stt_seconds: float = 0.0
    tts_chars: int = 0
    llm_tokens: int = 0
    turn_count: int = 0

    # Guardrails 3-strike policy counter (hybrid: in-memory cache + Redis persistence)
    harmful_attempt_count: int = 0

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since session start."""
        ref = self.end_time or datetime.now(timezone.utc)
        return (ref - self.start_time).total_seconds()

    @property
    def is_active(self) -> bool:
        """True if session has not yet ended."""
        return self.state not in (SessionState.ENDED, SessionState.TIMED_OUT)

    @property
    def is_timed_out(self) -> bool:
        """True if elapsed time exceeds SESSION_TIMEOUT_SECONDS."""
        return self.is_active and self.elapsed_seconds > SESSION_TIMEOUT_SECONDS

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict. Booleans are native bool, not strings."""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "call_type": self.call_type,
            "state": self.state.value,
            "language": self.language,
            "is_code_mixed": bool(self.is_code_mixed),     # native bool, not "True"
            "is_active": bool(self.is_active),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "outcome": self.outcome,
            "stt_seconds": round(self.stt_seconds, 2),
            "tts_chars": self.tts_chars,
            "llm_tokens": self.llm_tokens,
            "turn_count": self.turn_count,
            "harmful_attempt_count": self.harmful_attempt_count,
            "last_error": self.last_error,
        }


class _SessionContextManager:
    """Async context manager returned by ``SessionManager.acquire()``."""

    def __init__(self, session_id: str, sessions: dict, lock: asyncio.Lock) -> None:
        self._session_id = session_id
        self._sessions = sessions
        self._lock = lock

    async def __aenter__(self) -> CallSession:
        await self._lock.acquire()
        session = self._sessions.get(self._session_id)
        if session is None:
            self._lock.release()
            raise SessionError(
                f"Session '{self._session_id}' not found.",
                session_id=self._session_id,
            )
        return session

    async def __aexit__(self, *_) -> None:
        if self._lock.locked():
            self._lock.release()


class SessionManager:
    """Thread-safe in-memory store for active voice call sessions with
    Redis-backed strike count persistence for the 3-strike guardrails policy.

    Uses a hybrid approach:
    * In-memory ``CallSession`` for fast access to all session state.
    * Redis ``session:{session_id}:strikes`` for cross-worker/restart durability
      of ``harmful_attempt_count``.

    Use ``get_session_manager()`` to obtain the process-level singleton.

    Args:
        max_sessions: Maximum concurrent sessions (new ones rejected beyond this).
        redis_host: Redis hostname for strike persistence.
        redis_port: Redis port for strike persistence.
    """

    def __init__(
        self,
        max_sessions: int = MAX_CONCURRENT_SESSIONS,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
    ) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        # Async Redis client for strike count persistence (cross-worker durability)
        self._async_redis: aioredis.Redis = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        logger.info(
            "SessionManager initialised (max_sessions=%d, redis=%s:%d)",
            max_sessions, redis_host, redis_port,
        )

    # ── Session lifecycle 

    async def create_session(
        self,
        tenant_id: str,
        session_id: str,
        call_type: str = "inbound",
    ) -> CallSession:
        """Create and register a new call session.

        Hydrates ``harmful_attempt_count`` from Redis if a prior session
        with the same ID exists (e.g., after worker restart or reconnect).

        Raises:
            SessionError: If session_id already exists or max sessions reached.
        """
        async with self._lock:
            if session_id in self._sessions:
                raise SessionError(
                    f"Session '{session_id}' already exists.",
                    session_id=session_id,
                )
            active = sum(1 for s in self._sessions.values() if s.is_active)
            if active >= self._max_sessions:
                raise SessionError(
                    f"Max concurrent sessions ({self._max_sessions}) reached.",
                    active_count=active,
                )
            session = CallSession(
                session_id=session_id,
                tenant_id=tenant_id,
                call_type=call_type,
            )
            # Hydrate strike count from Redis (fail-open: 0 on error)
            session.harmful_attempt_count = await self._load_strikes(session_id)
            self._sessions[session_id] = session
            logger.info(
                "Session created: id=%s tenant=%s type=%s active=%d strikes=%d",
                session_id, tenant_id, call_type, active + 1,
                session.harmful_attempt_count,
            )
            return session

    async def get_session(self, session_id: str) -> CallSession:
        """Retrieve a session by ID.

        Raises:
            SessionError: If session is not found.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError(
                f"Session '{session_id}' not found.",
                session_id=session_id,
            )
        return session

    def acquire(self, session_id: str) -> _SessionContextManager:
        """Return an async context manager that holds the session lock for safe updates.

        Example::
            async with sm.acquire(session_id) as session:
                session.stt_seconds += duration
                session.turn_count += 1
        """
        return _SessionContextManager(session_id, self._sessions, self._lock)

    async def update_session(self, session_id: str, **kwargs: Any) -> CallSession:
        """Bulk-update mutable fields on a session.

        Supported kwargs: ``state``, ``language``, ``is_code_mixed``,
        ``stt_seconds`` (accumulated), ``tts_chars`` (accumulated),
        ``llm_tokens`` (accumulated), ``turn_count`` (incremented by kwarg value),
        ``harmful_attempt_count`` (direct set), ``last_error``, ``outcome``.

        Raises:
            SessionError: If session is not found.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError(
                    f"Session '{session_id}' not found.", session_id=session_id
                )
            for key, value in kwargs.items():
                if key in ("stt_seconds", "tts_chars", "llm_tokens", "turn_count"):
                    current = getattr(session, key, 0)
                    setattr(session, key, current + value)
                elif hasattr(session, key):
                    setattr(session, key, value)
                else:
                    logger.debug("update_session: unknown field '%s' ignored", key)
        return session

    async def end_session(
        self,
        session_id: str,
        outcome: str = "completed",
    ) -> Optional[CallSession]:
        """Mark session as ended and record outcome.

        Cleans up the Redis strike key on termination outcomes to prevent
        stale data if the same session_id is reused.

        Args:
            session_id: Session to end.
            outcome: ``"completed"``, ``"escalated"``, ``"failed"``,
                     ``"timed_out"``, ``"terminated"``.

        Returns:
            Final CallSession state for reporting, or None if not found.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.state = SessionState.ENDED
            session.outcome = outcome
            session.end_time = datetime.now(timezone.utc)

        logger.info(
            "Session ended: id=%s tenant=%s outcome=%s elapsed=%.1fs turns=%d strikes=%d",
            session_id, session.tenant_id, outcome,
            session.elapsed_seconds, session.turn_count,
            session.harmful_attempt_count,
        )

        # Clean up Redis strike key on terminal outcomes to prevent stale state
        if outcome in ("terminated", "completed"):
            await self._delete_strikes(session_id)

        return session

    def list_active_sessions(self, tenant_id: Optional[str] = None) -> list[dict]:
        """Return summary of all active sessions, optionally filtered by tenant."""
        sessions = [
            s.to_dict()
            for s in self._sessions.values()
            if s.is_active and (tenant_id is None or s.tenant_id == tenant_id)
        ]
        return sessions

    # ── Strike count persistence (Redis) 

    async def load_session_strikes(self, session_id: str) -> int:
        """Public wrapper to load strike count from Redis.

        Used by WebSocket handler per-turn to re-hydrate from Redis
        before graph invocation (handles cross-worker scenarios).
        """
        return await self._load_strikes(session_id)

    async def save_session_strikes(self, session_id: str, count: int) -> None:
        """Persist strike count to Redis with TTL.

        Called by WebSocket handler after every graph invocation
        where harmful_attempt_count may have changed.
        """
        try:
            await self._async_redis.set(
                _session_strike_key(session_id),
                str(count),
                ex=_SESSION_STRIKE_TTL_SECONDS,
            )
            logger.debug(
                "Strikes saved: session=%s count=%d ttl=%ds",
                session_id, count, _SESSION_STRIKE_TTL_SECONDS,
            )
        except Exception as exc:
            logger.error(
                "Failed to save strikes for session=%s (non-fatal): %s",
                session_id, exc,
            )

    # ── Private helpers 

    async def _load_strikes(self, session_id: str) -> int:
        """Internal: load strike count from Redis. Fail-open to 0."""
        try:
            val = await self._async_redis.get(_session_strike_key(session_id))
            if val is not None:
                count = int(val)
                logger.debug(
                    "Strikes loaded: session=%s count=%d", session_id, count
                )
                return count
        except Exception as exc:
            logger.warning(
                "Failed to load strikes for session=%s (fail-open to 0): %s",
                session_id, exc,
            )
        return 0

    async def _delete_strikes(self, session_id: str) -> None:
        """Internal: delete strike key from Redis (cleanup on session end)."""
        try:
            await self._async_redis.delete(_session_strike_key(session_id))
            logger.debug("Strikes deleted: session=%s", session_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete strikes for session=%s (non-fatal): %s",
                session_id, exc,
            )

    async def close(self) -> None:
        """Close async Redis connection gracefully (call in lifespan shutdown)."""
        await self._async_redis.aclose()
        logger.info("SessionManager Redis connection closed.")


# ── Process-level singleton 

_singleton: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return (or create) the process-level SessionManager singleton."""
    global _singleton
    if _singleton is None:
        _singleton = SessionManager()
    return _singleton