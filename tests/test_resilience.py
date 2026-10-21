import asyncio
import logging
import os
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI

from schemas import GradeRequest, GradeResponse
from services.embeddings import EmbeddingServiceError
from services.grader import Grader
from services.resilience import (
    CircuitBreaker,
    FallbackSettings,
    ModelUnavailableError,
    ResilientEmbeddingService,
)


class FakeEmbeddingService:
    def __init__(self, value: float = 0.6, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls = 0
        self.model_name = "fake-model"
        self.user_ids: list[str | None] = []

    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        self.calls += 1
        self.user_ids.append(user_id)
        if self.error:
            raise self.error
        return self.value


class SequencedEmbeddingService(FakeEmbeddingService):
    def __init__(self, failures: int, value: float = 0.7) -> None:
        super().__init__(value=value)
        self.failures = failures

    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        self.calls += 1
        self.user_ids.append(user_id)
        if self.failures:
            self.failures -= 1
            try:
                raise TimeoutError("inference timed out")
            except TimeoutError as exc:
                raise EmbeddingServiceError("embedding failed") from exc
        return self.value


class BadRequestError(EmbeddingServiceError):
    status_code = 400


class RateLimitError(EmbeddingServiceError):
    status_code = 429


def settings(
    *,
    enabled: bool = True,
    fallback_model: str | None = "backup-model",
    max_retries: int = 2,
    request_timeout_seconds: float = 1,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_cooldown_seconds: float = 30,
) -> FallbackSettings:
    return FallbackSettings(
        enabled=enabled,
        fallback_model=fallback_model,
        max_retries=max_retries,
        request_timeout_seconds=request_timeout_seconds,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=circuit_breaker_cooldown_seconds,
    )


class ResilienceTests(unittest.TestCase):
    def test_circuit_breaker_allows_probe_after_cooldown(self) -> None:
        breaker = CircuitBreaker(threshold=1, cooldown_seconds=30)
        with patch("services.resilience.time.monotonic", return_value=100):
            self.assertTrue(breaker.record_failure())
            self.assertFalse(breaker.allow_request())
        with patch("services.resilience.time.monotonic", return_value=131):
            self.assertTrue(breaker.allow_request())

    def test_normal_path_calls_primary_once_with_same_user_scope(self) -> None:
        primary = FakeEmbeddingService(value=0.82)
        fallback = FakeEmbeddingService(value=0.2)
        service = ResilientEmbeddingService(
            primary, lambda: fallback, settings(), sleep=lambda _: None
        )

        self.assertEqual(service.similarity("reference", "candidate", "student-1"), 0.82)
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(primary.user_ids, ["student-1"])

    def test_retries_transient_error_then_returns_primary_result(self) -> None:
        primary = SequencedEmbeddingService(failures=2)
        fallback = FakeEmbeddingService()
        sleeps: list[float] = []
        service = ResilientEmbeddingService(
            primary,
            lambda: fallback,
            settings(),
            sleep=sleeps.append,
            jitter=lambda low, high: high,
        )

        self.assertEqual(service.similarity("reference", "candidate"), 0.7)
        self.assertEqual(primary.calls, 3)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_uses_fallback_after_primary_retry_budget_exhausted(self) -> None:
        primary = SequencedEmbeddingService(failures=3)
        fallback = FakeEmbeddingService(value=0.35)
        service = ResilientEmbeddingService(
            primary, lambda: fallback, settings(), sleep=lambda _: None
        )

        self.assertEqual(service.similarity("reference", "candidate"), 0.35)
        self.assertEqual(primary.calls, 3)
        self.assertEqual(fallback.calls, 1)

    def test_does_not_retry_bad_request_or_use_fallback(self) -> None:
        primary = FakeEmbeddingService(error=BadRequestError("invalid input"))
        fallback = FakeEmbeddingService()
        service = ResilientEmbeddingService(
            primary, lambda: fallback, settings(), sleep=lambda _: None
        )

        with self.assertRaisesRegex(BadRequestError, "invalid input"):
            service.similarity("reference", "candidate")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 0)

    def test_retries_rate_limit_errors(self) -> None:
        primary = FakeEmbeddingService(error=RateLimitError("rate limited"))
        fallback = FakeEmbeddingService(value=0.35)
        service = ResilientEmbeddingService(
            primary, lambda: fallback, settings(), sleep=lambda _: None
        )

        self.assertEqual(service.similarity("reference", "candidate"), 0.35)
        self.assertEqual(primary.calls, 3)
        self.assertEqual(fallback.calls, 1)

    def test_primary_circuit_opens_and_uses_backup_without_calling_primary(self) -> None:
        primary = SequencedEmbeddingService(failures=10)
        fallback = FakeEmbeddingService(value=0.4)
        service = ResilientEmbeddingService(
            primary,
            lambda: fallback,
            settings(max_retries=0, circuit_breaker_threshold=2),
            sleep=lambda _: None,
        )

        self.assertEqual(service.similarity("r", "c"), 0.4)
        self.assertEqual(service.similarity("r", "c"), 0.4)
        self.assertEqual(service.similarity("r", "c"), 0.4)
        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 3)

    def test_timeout_falls_back_and_failure_is_not_cached_by_caller(self) -> None:
        release = threading.Event()

        class SlowModel(FakeEmbeddingService):
            def similarity(self, reference: str, candidate: str, user_id=None) -> float:
                self.calls += 1
                release.wait(timeout=1)
                return 0.9

        primary = SlowModel()
        fallback = FakeEmbeddingService(value=0.3)
        service = ResilientEmbeddingService(
            primary,
            lambda: fallback,
            settings(max_retries=0, request_timeout_seconds=0.02),
            sleep=lambda _: None,
        )
        try:
            self.assertEqual(service.similarity("r", "c"), 0.3)
            self.assertEqual(primary.calls, 1)
        finally:
            release.set()

    def test_returns_clear_error_when_backup_is_not_configured(self) -> None:
        primary = SequencedEmbeddingService(failures=1)
        service = ResilientEmbeddingService(
            primary,
            None,
            settings(fallback_model=None, max_retries=0),
            sleep=lambda _: None,
        )

        with self.assertRaises(ModelUnavailableError):
            service.similarity("reference", "candidate")


class FallbackResponseTests(unittest.TestCase):
    def test_startup_remains_available_if_primary_model_cannot_load(self) -> None:
        from main import lifespan
        from services.embeddings import SentenceTransformerEmbeddingService

        app = FastAPI()
        with (
            patch.dict(
                os.environ,
                {
                    "FALLBACK_ENABLED": "true",
                    "FALLBACK_MODEL": "",
                    "CACHE_ENABLED": "false",
                },
            ),
            patch.object(
                SentenceTransformerEmbeddingService,
                "from_environment",
                side_effect=EmbeddingServiceError("model unavailable"),
            ),
        ):
            async def run_lifespan() -> None:
                async with lifespan(app):
                    self.assertIsNone(app.state.grader)
                    self.assertFalse(app.state.embedding_ready)

            asyncio.run(run_lifespan())

    def test_sqlite_audit_failure_does_not_escape_logger(self) -> None:
        from logger import _SQLiteAuditHandler

        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "not-a-database-file"
            database_path.mkdir()
            with patch.dict(
                os.environ,
                {
                    "CIRCUIT_BREAKER_THRESHOLD": "2",
                    "CIRCUIT_BREAKER_COOLDOWN_SECONDS": "30",
                },
            ):
                handler = _SQLiteAuditHandler(database_path)
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="grading",
                args=(),
                exc_info=None,
            )
            record.audit_data = {"event_type": "grading"}

            handler.emit(record)
            handler.emit(record)

    def test_unavailable_response_keeps_schema_and_marks_placeholder_score(self) -> None:
        from main import _unavailable_grade

        request = GradeRequest(
            question="What is evaporation?",
            model_answer="Liquid changes into gas.",
            student_answer="A liquid becomes gas.",
            max_score=5,
        )
        response = _unavailable_grade(request)

        self.assertEqual(set(GradeResponse.model_fields), set(response.model_dump()))
        self.assertEqual(response.score, 0)
        self.assertIn("temporarily unavailable", response.feedback_ar)
        self.assertTrue(response.review_flags)

    def test_unavailable_response_uses_arabic_feedback_for_arabic_input(self) -> None:
        from main import _unavailable_grade

        request = GradeRequest(
            question="ما هو التبخر؟",
            model_answer="تحول السائل إلى غاز.",
            student_answer="يتحول السائل إلى غاز.",
            max_score=5,
        )

        response = _unavailable_grade(request)

        self.assertIn("تعذر التقييم مؤقتًا", response.feedback_ar)

    def test_grade_route_returns_final_fallback_in_normal_response_shape(self) -> None:
        from main import grade as grade_route

        primary = SequencedEmbeddingService(failures=1)
        resilient = ResilientEmbeddingService(
            primary,
            None,
            settings(max_retries=0, fallback_model=None),
            sleep=lambda _: None,
        )
        fake_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    grader=Grader(resilient),
                    fallback_enabled=True,
                )
            )
        )
        payload = GradeRequest(
            question="ما وظيفة الجذور؟",
            model_answer="تمتص الماء.",
            student_answer="تمتص الماء.",
            max_score=5,
        )

        response = grade_route(payload, fake_request)

        self.assertEqual(set(GradeResponse.model_fields), set(response.model_dump()))
        self.assertEqual(response.score, 0)
        self.assertTrue(response.review_flags)


if __name__ == "__main__":
    unittest.main()
