"""
Centralized logging configuration for KoyalAI.
"""

import logging
import logging.handlers
import os
from pathlib import Path

LOG_DIR = Path(os.getenv("KOYAL_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "koyalai.log"


def setup_logging() -> None:
    """Configure root logger with rotating file + console handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on reload
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)-30s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # File handler: DEBUG+, 10MB rotation, 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Console handler: INFO+ only (keeps terminal clean)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "livekit", "livekit.agents", "websockets", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Logging configured: file=%s (DEBUG+), console (INFO+)", LOG_FILE)


def get_log_path() -> Path:
    return LOG_FILE