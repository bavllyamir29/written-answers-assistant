import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from schemas import GradeRequest, GradeResponse
from services.embeddings import EmbeddingServiceError, SentenceTransformerEmbeddingService
from services.grader import Grader

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    try:
        embedding_service = SentenceTransformerEmbeddingService.from_environment(model_name)
        app.state.grader = Grader(embedding_service)
    except EmbeddingServiceError:
        logger.exception("Could not initialize the embedding service")
        raise
    yield


app = FastAPI(
    title="Arabic Short-Answer Grader",
    description="Hybrid grading support for Arabic short-answer and descriptive responses.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_service": "ready" if hasattr(request.app.state, "grader") else "not_ready",
    }


@app.post("/grade", response_model=GradeResponse)
def grade(payload: GradeRequest, request: Request) -> GradeResponse:
    grader: Grader | None = getattr(request.app.state, "grader", None)
    if grader is None:
        raise HTTPException(status_code=503, detail="The grading service is not ready.")

    try:
        return grader.grade(payload)
    except EmbeddingServiceError as exc:
        logger.exception("Embedding service failed while grading")
        raise HTTPException(
            status_code=503,
            detail="Semantic analysis is temporarily unavailable. Please try again later.",
        ) from exc
