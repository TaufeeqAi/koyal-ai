"""
Importing this package registers all Prometheus metric instruments at
module load time. FastAPI lifespan and any background tasks that import
from this package benefit from having metrics registered before the
first request arrives.

Public surface:
    METRICS          — typed _KoyalMetrics dataclass singleton
    langfuse_client  — Langfuse v4 singleton + NoOp fallback
    instrumented_graph — zero-touch LangGraph wrapper
"""

from backend.observability.prometheus_metrics import METRICS  
from backend.observability.langfuse_client import (           
    init_langfuse,
    get_langfuse_client,
    make_callback_handler,
    flush as flush_langfuse,
    is_langfuse_available,
)

__all__ = [
    "METRICS",
    "init_langfuse",
    "get_langfuse_client",
    "make_callback_handler",
    "flush_langfuse",
    "is_langfuse_available",
]