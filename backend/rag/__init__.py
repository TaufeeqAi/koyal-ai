"""
KoyalAI RAG sub-package.

Public surface:
    BilingualChunker        — Devanagari-aware text splitter
    MultilingualEmbedder    — LaBSE 768-dim encoder
    MultilingualIngestor    — Qdrant per-tenant ingestion pipeline
    MultilingualRetriever   — Cross-lingual ANN + reranking retriever
"""

from backend.rag.chunker import BilingualChunker
from backend.rag.embedder import MultilingualEmbedder
from backend.rag.ingestor import MultilingualIngestor
from backend.rag.retriever import MultilingualRetriever

__all__ = [
    "BilingualChunker",
    "MultilingualEmbedder",
    "MultilingualIngestor",
    "MultilingualRetriever",
]