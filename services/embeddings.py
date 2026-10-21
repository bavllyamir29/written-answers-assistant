import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from services.embedding_protocol import EmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when an embedding provider cannot load or encode text."""


class SentenceTransformerEmbeddingService:
    def __init__(self, model: SentenceTransformer, model_name: str) -> None:
        self._model = model
        self.model_name = model_name

    @classmethod
    def from_environment(cls, default_model: str) -> "SentenceTransformerEmbeddingService":
        model_name = os.getenv("EMBEDDING_MODEL", default_model)
        return cls.from_model_name(model_name)

    @classmethod
    def from_model_name(cls, model_name: str) -> "SentenceTransformerEmbeddingService":
        device = os.getenv("EMBEDDING_DEVICE")
        try:
            model = SentenceTransformer(model_name, device=device or None)
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Could not load embedding model '{model_name}'."
            ) from exc
        return cls(model, model_name)

    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        try:
            vectors = self._model.encode(
                [reference, candidate],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            cosine = float(np.dot(vectors[0], vectors[1]))
        except Exception as exc:
            raise EmbeddingServiceError("Could not calculate semantic similarity.") from exc

        if not np.isfinite(cosine):
            raise EmbeddingServiceError("The embedding model returned a non-finite similarity.")
        LOW, HIGH = 0.45, 0.85
        score = (cosine - LOW) / (HIGH - LOW)
        return min(1.0, max(0.0, score))