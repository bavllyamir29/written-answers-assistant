from typing import Protocol


class EmbeddingService(Protocol):
    def similarity(self, reference: str, candidate: str) -> float:
        """Return cosine similarity normalized to [0, 1]."""
