from __future__ import annotations

import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv

from backend.exceptions import ConfigValidationError,TenantNotFoundError

load_dotenv()

logger = logging.getLogger(__name__)

# ── API Keys 
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_EVAL_API_KEY = os.getenv("GROQ_EVAL_API_KEY")
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

# ── LiveKit 
LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_WS_URL: str = os.getenv("LIVEKIT_WS_URL", "ws://localhost:7880")
LIVEKIT_SIP_TRUNK_ID: str = os.getenv("LIVEKIT_SIP_TRUNK_ID", "trunk_in_1")

# ── Infrastructure 
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
REDIS_DECODE_RESPONSES: bool = True

# ── Language 
DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "hi-IN")
SUPPORTED_LANGUAGES: list[str] = os.getenv(
    "SUPPORTED_LANGUAGES", "hi-IN,en-IN,mr-IN,ta-IN,te-IN,kn-IN,bn-IN"
).split(",")

# ── Embedding — LaBSE 
EMBEDDING_MODEL: str = "sentence-transformers/LaBSE"
EMBEDDING_DIMENSION: int = 768
EMBEDDING_BATCH_SIZE: int = 16  # CPU-safe default; override at runtime if needed

# ── ReRanker 
RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ── Qdrant
QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "20"))
QDRANT_QUERY_MAX_RETRIES: int = int(
    os.getenv("QDRANT_QUERY_MAX_RETRIES", "3")
)

QDRANT_QUERY_BACKOFF_SECONDS: float = float(
    os.getenv("QDRANT_QUERY_BACKOFF_SECONDS", "0.5")
)

# ── Chunking 
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 40

# ── Retrieval 
TOP_K_RETRIEVAL: int = 10
RERANK_TOP_K: int = 3
SCORE_THRESHOLD: float = 0.15  # Lower than monolingual due to cross-lingual gap

# ── Ingestion Resilience 
UPSERT_BATCH_SIZE: int = 100
UPSERT_MAX_RETRIES: int = 3
UPSERT_RETRY_DELAY_SECONDS: float = 2.0

# ── Data + Tenants 
DATA_DIR: Path = Path(__file__).parent.parent / "data"
TENANTS: list[str] = ["tenant_hdfc_bank", "tenant_swiggy_support"]


# ── LLM Config (Groq) 

GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0"))
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TIMEOUT: int = int(os.getenv("GROQ_TIMEOUT", "30"))

# ── LLM Config (Gemini) 

GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0"))
GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "512"))
GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))

# ── Sarvam API Endpoints + Timeouts
SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL: str = "https://api.sarvam.ai/translate"
SARVAM_LID_URL: str = "https://api.sarvam.ai/text-lid"
SARVAM_TIMEOUT: int = int(os.getenv("SARVAM_TIMEOUT", "5"))    # seconds
SARVAM_MAX_RETRIES: int = int(os.getenv("SARVAM_MAX_RETRIES", "3"))
SARVAM_BACKOFF_BASE: float = float(os.getenv("SARVAM_BACKOFF_BASE", "2.0"))

# ── Sarvam Voice Mapping 
SARVAM_VOICE_MAP: dict[str, str] = {
    "hi-IN": "meera",
    "en-IN": "anushka",
    "hi-IN+en-IN": "meera",
}

# ── Safety Thresholds 

# Cosine similarity threshold for semantic emergency detection via LaBSE
EMERGENCY_SEMANTIC_THRESHOLD: float = float(
    os.getenv("EMERGENCY_SEMANTIC_THRESHOLD", "0.80")
)

# Agent Pipeline Config

AGENT_MAX_RETRIES: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
AGENT_BACKOFF_BASE: float = float(os.getenv("AGENT_BACKOFF_BASE", "2.0"))

# Verification: minimum faithfulness score to mark response as verified
VERIFICATION_SCORE_THRESHOLD: float = float(
    os.getenv("VERIFICATION_SCORE_THRESHOLD", "0.7")
)

# ── NeMo Guardrails
GUARDRAILS_CONFIG_PATH: Path = Path(
    os.getenv("GUARDRAILS_CONFIG_PATH", str(Path(__file__).parent / "safety" / "guardrails_config"))
)
GUARDRAILS_ENABLED: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() in ("true", "1", "yes")


# ── STT 
STT_MODEL: str = os.getenv("STT_MODEL", "saaras:v2")
STT_TIMEOUT: int = int(os.getenv("STT_TIMEOUT", "15"))
STT_MAX_RETRIES: int = int(os.getenv("STT_MAX_RETRIES", "3"))
STT_BACKOFF_BASE: float = float(os.getenv("STT_BACKOFF_BASE", "2.0"))
STT_CONFIDENCE_THRESHOLD: float = float(os.getenv("STT_CONFIDENCE_THRESHOLD", "0.4"))

# ── TTS 
TTS_MODEL: str = os.getenv("TTS_MODEL", "bulbul:v1")
TTS_TIMEOUT: int = int(os.getenv("TTS_TIMEOUT", "10"))
TTS_MAX_RETRIES: int = int(os.getenv("TTS_MAX_RETRIES", "3"))
TTS_BACKOFF_BASE: float = float(os.getenv("TTS_BACKOFF_BASE", "2.0"))
TTS_PACE: float = float(os.getenv("TTS_PACE", "1.0"))
TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "16000"))
TTS_MAX_CHARS_PER_CHUNK: int = int(os.getenv("TTS_MAX_CHARS_PER_CHUNK", "500"))

# ── VAD 
VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
# 0=permissive → 3=aggressive. 2 recommended for telephony.
VAD_SAMPLE_RATE: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
VAD_FRAME_DURATION_MS: int = int(os.getenv("VAD_FRAME_DURATION_MS", "30"))
# WebRTC VAD accepts: 10ms, 20ms, or 30ms frames only.
VAD_SPEECH_THRESHOLD: int = int(os.getenv("VAD_SPEECH_THRESHOLD", "8"))
# Consecutive voiced frames before declaring speech start.
VAD_SILENCE_THRESHOLD: int = int(os.getenv("VAD_SILENCE_THRESHOLD", "25"))
# Consecutive silent frames before declaring utterance end (25 × 30ms = 750ms).
VAD_MIN_SPEECH_FRAMES: int = int(os.getenv("VAD_MIN_SPEECH_FRAMES", "5"))
# Minimum voiced frames to count as utterance (filters noise bursts).
VAD_ENERGY_THRESHOLD: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "200.0"))
# RMS energy below this → forced silence (energy guard secondary to webrtcvad).

# ── Session 
SESSION_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", "300"))
MAX_CONCURRENT_SESSIONS: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "100"))

# ── Outbound 
OUTBOUND_MAX_CONCURRENT: int = int(os.getenv("OUTBOUND_MAX_CONCURRENT", "5"))

# ── Cost rates (₹) 
STT_COST_PER_MINUTE: float = float(os.getenv("STT_COST_PER_MINUTE", "0.50"))
TTS_COST_PER_CHAR: float = float(os.getenv("TTS_COST_PER_CHAR", "0.0015"))
LLM_COST_PER_TOKEN: float = float(os.getenv("LLM_COST_PER_TOKEN", "0.0"))
COST_KEY_TTL_DAYS: int = int(os.getenv("COST_KEY_TTL_DAYS", "30"))
STT_COST_PER_SECOND = STT_COST_PER_MINUTE / 60.0

# ── WebSocket 
WS_RECEIVE_TIMEOUT: float = float(os.getenv("WS_RECEIVE_TIMEOUT", "0.1"))
WS_GREETING_ENABLED: bool = os.getenv("WS_GREETING_ENABLED", "true").lower() == "true"



def load_tenant_config(tenant_id: str) -> dict:
    """Load and return the config.json for a given tenant.

    Args:
        tenant_id: Tenant directory name, e.g. ``"tenant_hdfc_bank"``.

    Returns:
        Parsed JSON dict with keys such as ``company_name``,
        ``primary_language``, ``cost_rates_inr``, etc.

    Raises:
        TenantNotFoundError: If the tenant directory or config.json is absent.
        ConfigValidationError: If required keys are missing from the config.

    Example:
        >>> cfg = load_tenant_config("tenant_hdfc_bank")
        >>> cfg["company_name"]
        'HDFC Bank'
    """
    config_path = DATA_DIR / tenant_id / "config.json"
    if not config_path.exists():
        raise TenantNotFoundError(
            f"Tenant config not found at: {config_path}",
            tenant_id=tenant_id,
        )
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    required_keys = ("tenant_id", "company_name", "primary_language")
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ConfigValidationError(
            f"Tenant config for '{tenant_id}' is missing required keys: {missing}",
            tenant_id=tenant_id,
            missing_keys=missing,
        )
    logger.debug("Loaded tenant config for '%s'", tenant_id)
    return cfg


def validate_runtime_config() -> None:
    """Check that all required secrets are present at startup.

    Raises:
        ConfigValidationError: If GROQ_API_KEY is absent.

    Example:
        >>> validate_runtime_config()   # raises if GROQ_API_KEY unset
    """
    errors: list[str] = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set — required for LLM calls.")
    if not SARVAM_API_KEY:
        # Warn rather than error — script analysis still works without it
        logger.warning(
            "SARVAM_API_KEY is not set. Sarvam LID + translation will fall back "
            "to script-based detection and passthrough respectively."
        )
    if errors:
        raise ConfigValidationError(
            "Runtime configuration invalid:\n" + "\n".join(f"  • {e}" for e in errors)
        )