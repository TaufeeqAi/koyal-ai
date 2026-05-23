from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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

# ── Sarvam API Endpoints 
SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL: str = "https://api.sarvam.ai/translate"
SARVAM_LID_URL: str = "https://api.sarvam.ai/text-lid"

# ── Sarvam Voice Mapping 
SARVAM_VOICE_MAP: dict[str, str] = {
    "hi-IN": "meera",
    "en-IN": "anushka",
    "hi-IN+en-IN": "meera",
}


def load_tenant_config(tenant_id: str) -> dict:
    """
    Load and parse the config.json for a tenant.

    Args:
        tenant_id: e.g. "tenant_hdfc_bank"

    Returns:
        Parsed dict with keys: tenant_id, company_name, primary_language,
        supported_languages, default_voice, cost_rates_inr, etc.

    Raises:
        FileNotFoundError: if config.json missing for tenant.
        json.JSONDecodeError: if config.json is malformed.
    """
    config_path = DATA_DIR / tenant_id / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Tenant config not found: {config_path}. "
            f"Ensure data/{tenant_id}/config.json exists."
        )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)