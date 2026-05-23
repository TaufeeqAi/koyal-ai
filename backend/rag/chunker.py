from __future__ import annotations

from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class BilingualChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter with Devanagari separators.

    Usage:
        chunker = BilingualChunker()
        chunks = chunker.chunk(raw_text, metadata={"tenant_id": "tenant_hdfc_bank"})
    """

    # Separator priority: section > line > danda (Hindi) > period (English)
    SEPARATORS: List[str] = [
        "\n\n",   # Section / paragraph break
        "\n",     # Line break
        "।",      # Devanagari danda — Hindi sentence terminator 
        ". ",     # English sentence end (space after period avoids URL splits)
        "? ",     # Question end
        "! ",     # Exclamation end
        ", ",     # Clause break
        " ",      # Word boundary (fallback)
    ]

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,             # character count, not token count
            is_separator_regex=False,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> List[dict[str, Any]]:
        """
        Split text into chunks and attach metadata to each.

        Args:
            text:     Raw document string (Hindi, English, or mixed).
            metadata: Dict attached to every chunk — should include at minimum
                      {"tenant_id": ..., "language": ..., "source": ...}

        Returns:
            List of dicts:
                {
                    "text": str,          # chunk content
                    "chunk_index": int,   # 0-based position in document
                    "char_count": int,    # character count of this chunk
                    "metadata": dict,     # pass-through metadata + chunk_index
                }
        """
        if not text or not text.strip():
            return []

        raw_chunks: List[str] = self._splitter.split_text(text)
        base_meta: dict[str, Any] = metadata or {}

        return [
            {
                "text": chunk.strip(),
                "chunk_index": idx,
                "char_count": len(chunk.strip()),
                "metadata": {**base_meta, "chunk_index": idx},
            }
            for idx, chunk in enumerate(raw_chunks)
            if chunk.strip()   # discard whitespace-only chunks
        ]