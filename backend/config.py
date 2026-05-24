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

# ── Language 
DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "hi-IN")
SUPPORTED_LANGUAGES: list[str] = os.getenv(
    "SUPPORTED_LANGUAGES", "hi-IN,en-IN,mr-IN,ta-IN,te-IN,kn-IN,bn-IN"
).split(",")

# ── Embedding — LaBSE 
EMBEDDING_MODEL: str = "sentence-transformers/LaBSE"
EMBEDDING_DIMENSION: int = 768
EMBEDDING_BATCH_SIZE: int = 16  # CPU-safe default; override at runtime if needed

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
        ConfigValidationError: If GROQ_API_KEY is absent (required for Phase 2).

    Example:
        >>> validate_runtime_config()   # raises if GROQ_API_KEY unset
    """
    errors: list[str] = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set — required for Phase 2 LLM calls.")
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