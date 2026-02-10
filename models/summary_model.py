"""Model for managing story chunks and hierarchical summaries."""

import json
import os
from base.observable import Observable
from typing import List, Dict


class SummaryModel(Observable):
    """Manages story chunking, summaries, and rolling compression."""

    def __init__(self):
        super().__init__()
        # List of dictionaries containing chunk metadata
        # Each chunk: {id, text, start_pos, end_pos, tokens, summary, summary_tokens}
        self._chunks: List[Dict] = []

        # Rolling summary that incorporates all chunks, kept under token limit
        self._rolling_summary: str = ""
        self._rolling_summary_tokens: int = 0

        # Track next chunk ID
        self._next_chunk_id: int = 0

        # Configuration
        self._target_chunk_size: int = 3000  # tokens
        self._max_rolling_summary_tokens: int = 1000

    @property
    def chunks(self) -> List[Dict]:
        """Get list of all chunks."""
        return self._chunks.copy()

    @property
    def rolling_summary(self) -> str:
        """Get current rolling summary."""
        return self._rolling_summary

    @property
    def rolling_summary_tokens(self) -> int:
        """Get token count of rolling summary."""
        return self._rolling_summary_tokens

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self._chunks)

    @property
    def summarized_chunks(self) -> int:
        """Get count of chunks with summaries."""
        return sum(1 for chunk in self._chunks if chunk.get("summary"))

    def add_chunk(self, text: str, start_pos: int, end_pos: int, tokens: int) -> int:
        """Add a new chunk to the model.

        Args:
            text: The chunk text content
            start_pos: Starting character position in full story
            end_pos: Ending character position in full story
            tokens: Estimated token count

        Returns:
            int: The chunk ID
        """
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1

        chunk = {
            "id": chunk_id,
            "text": text,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "tokens": tokens,
            "summary": None,
            "summary_tokens": 0,
        }

        self._chunks.append(chunk)
        self.notify_observers("chunk_added", chunk)
        return chunk_id

    def set_chunk_summary(self, chunk_id: int, summary: str, summary_tokens: int):
        """Set the summary for a specific chunk.

        Args:
            chunk_id: ID of the chunk to update
            summary: Summary text
            summary_tokens: Estimated token count of summary
        """
        for chunk in self._chunks:
            if chunk["id"] == chunk_id:
                chunk["summary"] = summary
                chunk["summary_tokens"] = summary_tokens
                self.notify_observers("chunk_summarized", chunk)
                return

    def update_rolling_summary(self, summary: str, tokens: int):
        """Update the rolling summary.

        Args:
            summary: New rolling summary text
            tokens: Estimated token count
        """
        self._rolling_summary = summary
        self._rolling_summary_tokens = tokens
        self.notify_observers(
            "rolling_summary_updated", {"summary": summary, "tokens": tokens}
        )

    def clear(self):
        """Clear all chunks and summaries."""
        self._chunks.clear()
        self._rolling_summary = ""
        self._rolling_summary_tokens = 0
        self._next_chunk_id = 0
        self.notify_observers("cleared", None)

    def get_context_for_llm(
        self, raw_recent_text: str, raw_recent_tokens: int
    ) -> tuple[str, int]:
        """Get the complete context to send to LLM.

        Args:
            raw_recent_text: Recent story text to include raw
            raw_recent_tokens: Token count of raw text

        Returns:
            tuple: (context_text, estimated_tokens)
        """
        if not self._rolling_summary:
            return raw_recent_text, raw_recent_tokens

        context_parts = []
        total_tokens = 0

        # Add rolling summary if it exists
        if self._rolling_summary:
            context_parts.append(
                f"STORY SUMMARY (prior events):\n{self._rolling_summary}"
            )
            total_tokens += self._rolling_summary_tokens

        # Add recent raw content
        if raw_recent_text:
            context_parts.append(f"\n\nRECENT EVENTS (verbatim):\n{raw_recent_text}")
            total_tokens += raw_recent_tokens

        return "\n".join(context_parts), total_tokens

    def save_to_file(self, filepath: str) -> bool:
        """Save chunks and summaries to a JSON file.

        Args:
            filepath: Path to save file

        Returns:
            bool: True if successful
        """
        try:
            data = {
                "chunks": self._chunks,
                "rolling_summary": self._rolling_summary,
                "rolling_summary_tokens": self._rolling_summary_tokens,
                "next_chunk_id": self._next_chunk_id,
                "target_chunk_size": self._target_chunk_size,
                "max_rolling_summary_tokens": self._max_rolling_summary_tokens,
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving summary data: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load chunks and summaries from a JSON file.

        Args:
            filepath: Path to load from

        Returns:
            bool: True if successful
        """
        try:
            if not os.path.exists(filepath):
                return False

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._chunks = data.get("chunks", [])
            self._rolling_summary = data.get("rolling_summary", "")
            self._rolling_summary_tokens = data.get("rolling_summary_tokens", 0)
            self._next_chunk_id = data.get("next_chunk_id", 0)
            self._target_chunk_size = data.get("target_chunk_size", 3000)
            self._max_rolling_summary_tokens = data.get(
                "max_rolling_summary_tokens", 1000
            )

            self.notify_observers("loaded", data)
            return True
        except Exception as e:
            print(f"Error loading summary data: {e}")
            return False
