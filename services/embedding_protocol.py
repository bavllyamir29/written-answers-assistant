from typing import Protocol


class EmbeddingService(Protocol):
    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        """Return cosine similarity normalized to [0, 1]."""
