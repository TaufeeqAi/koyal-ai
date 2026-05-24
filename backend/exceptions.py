class KoyalAIError(Exception):
    """Base exception for all KoyalAI domain errors."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


# ── External Service Errors 

class SarvamAPIError(KoyalAIError):
    """Raised when a Sarvam AI API call fails (non-2xx, timeout, parse error)."""


class GroqAPIError(KoyalAIError):
    """Raised when a Groq LLM API call fails."""


class QdrantError(KoyalAIError):
    """Raised when a Qdrant vector-store operation fails."""


# ── Configuration / Tenant Errors 

class TenantNotFoundError(KoyalAIError):
    """Raised when a tenant_id has no directory or config.json."""


class ConfigValidationError(KoyalAIError):
    """Raised when a required config value is missing or invalid."""


# ── Agent / Pipeline Errors 

class LanguageDetectionError(KoyalAIError):
    """Raised when language detection fails irrecoverably."""


class TranslationError(KoyalAIError):
    """Raised when translation via Sarvam Mayura fails irrecoverably."""


class RetrievalError(KoyalAIError):
    """Raised when Qdrant retrieval fails for a given tenant collection."""


class ResponseGenerationError(KoyalAIError):
    """Raised when the LLM fails to produce a response."""


class VerificationError(KoyalAIError):
    """Raised when Chain-of-Verification encounters an unrecoverable error."""


class EmergencyDetectionError(KoyalAIError):
    """Raised when emergency detection fails to initialise (model load error)."""


class GuardrailsError(KoyalAIError):
    """Raised when guardrails processing encounters an unrecoverable error."""