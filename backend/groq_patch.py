"""Strip stream_usage from ChatGroq API params — NeMo Guardrails hardcodes it."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _apply_patch() -> None:
    try:
        from langchain_groq.chat_models import ChatGroq

        _orig = ChatGroq._create_message_dicts

        def _patched(self, messages: list, stop: list[str] | None = None):
            message_dicts, params = _orig(self, messages, stop)
            params.pop("stream_usage", None)
            return message_dicts, params

        ChatGroq._create_message_dicts = _patched
        logger.info("groq_patch: applied")

    except Exception as exc:
        logger.warning("groq_patch: failed — %s", exc)


_apply_patch()