import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from schemas import GradeRequest, GradeResponse
from services.embeddings import EmbeddingServiceError, SentenceTransformerEmbeddingService
from services.embedding_cache import RedisCachedEmbeddingService
from services.grader import Grader
from services.resilience import (
    FallbackSettings,
    ModelUnavailableError,
    ResilientEmbeddingService,
)

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    try:
        settings = FallbackSettings.from_environment()
        using_startup_fallback = False
        primary_embedding_service: SentenceTransformerEmbeddingService | None = None
        try:
            primary_embedding_service = SentenceTransformerEmbeddingService.from_environment(
                model_name
            )
        except EmbeddingServiceError as primary_error:
            if not settings.enabled:
                raise
            logger.exception(
                "Fallback event dependency=primary_model outcome=startup_failure"
            )
            if not settings.fallback_model:
                logger.error(
                    "Fallback event dependency=fallback_model "
                    "outcome=unavailable fallback_model=not_configured"
                )
            else:
                try:
                    primary_embedding_service = (
                        SentenceTransformerEmbeddingService.from_model_name(
                            settings.fallback_model
                        )
                    )
                    using_startup_fallback = True
                    logger.warning(
                        "Fallback event dependency=fallback_model outcome=used "
                        "model=%s after_primary_error=%s",
                        settings.fallback_model,
                        primary_error,
                    )
                except EmbeddingServiceError:
                    logger.exception(
                        "Fallback event dependency=fallback_model "
                        "outcome=startup_failure model=%s",
                        settings.fallback_model,
                    )
        if primary_embedding_service is None:
            app.state.grader = None
            app.state.fallback_enabled = settings.enabled
            app.state.embedding_ready = False
            yield
            return
        fallback_factory = None
        if settings.enabled and settings.fallback_model and not using_startup_fallback:
            fallback_factory = lambda: SentenceTransformerEmbeddingService.from_model_name(
                settings.fallback_model
            )
        embedding_service = ResilientEmbeddingService(
            primary_embedding_service,
            fallback_factory,
            settings,
        )
        embedding_service = RedisCachedEmbeddingService.from_environment(
            embedding_service,
            circuit_breaker_threshold=settings.circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
            fallback_enabled=settings.enabled,
        )
        app.state.grader = Grader(embedding_service)
        app.state.fallback_enabled = settings.enabled
        app.state.embedding_ready = True
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
    ready = getattr(request.app.state, "embedding_ready", False)
    return {
        "status": "ok" if ready else "degraded",
        "embedding_service": "ready" if ready else "not_ready",
    }


@app.post("/grade", response_model=GradeResponse)
def grade(payload: GradeRequest, request: Request) -> GradeResponse:
    grader: Grader | None = getattr(request.app.state, "grader", None)
    if grader is None:
        if getattr(request.app.state, "fallback_enabled", True):
            logger.error(
                "Fallback event dependency=model outcome=final_response "
                "error=all_models_unavailable"
            )
            return _unavailable_grade(payload)
        raise HTTPException(status_code=503, detail="The grading service is not ready.")

    try:
        return grader.grade(payload)
    except (EmbeddingServiceError, ModelUnavailableError) as exc:
        logger.error("Fallback event dependency=model outcome=final_response error=%s", exc)
        if getattr(request.app.state, "fallback_enabled", True):
            return _unavailable_grade(payload)
        raise HTTPException(
            status_code=503,
            detail="Semantic analysis is temporarily unavailable. Please try again later.",
        ) from exc


def _unavailable_grade(payload: GradeRequest) -> GradeResponse:
    user_text = f"{payload.question} {payload.student_answer}"
    arabic_chars = sum("\u0600" <= character <= "\u06ff" for character in user_text)
    latin_chars = sum(character.isascii() and character.isalpha() for character in user_text)
    if arabic_chars >= latin_chars:
        feedback = "تعذر التقييم مؤقتًا بسبب مشكلة في الخدمة. يرجى المحاولة بعد قليل؛ لم تُحتسب هذه الدرجة."
    else:
        feedback = (
            "The grading service is temporarily unavailable. Please try again shortly; "
            "this is not an awarded grade."
        )
    return GradeResponse(
        score=0,
        max_score=payload.max_score,
        final_ratio=0,
        base_final_ratio=0,
        semantic_similarity=0,
        keyword_match_ratio=0,
        completeness_ratio=0,
        matched_keywords=[],
        missing_keywords=[],
        feedback_ar=feedback,
        review_flags=["لم تكتمل عملية التقييم؛ لا تعتمد الدرجة حتى إعادة المحاولة."]
        if arabic_chars >= latin_chars
        else ["Grading did not complete; do not use this score until grading is retried."],
        contradiction_cap_applied=False,
    )
