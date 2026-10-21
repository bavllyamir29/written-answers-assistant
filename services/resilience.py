import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from services.embedding_protocol import EmbeddingService

logger = logging.getLogger("uvicorn.error")
T = TypeVar("T")


@dataclass(frozen=True)
class FallbackSettings:
    enabled: bool = True
    fallback_model: str | None = None
    max_retries: int = 2
    request_timeout_seconds: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "FallbackSettings":
        enabled = os.getenv("FALLBACK_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        max_retries = int(os.getenv("MAX_RETRIES", "2"))
        request_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
        threshold = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
        cooldown = float(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30"))
        if not 0 <= max_retries <= 2:
            raise ValueError("MAX_RETRIES must be between 0 and 2.")
        if request_timeout <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than zero.")
        if threshold <= 0:
            raise ValueError("CIRCUIT_BREAKER_THRESHOLD must be greater than zero.")
        if cooldown <= 0:
            raise ValueError("CIRCUIT_BREAKER_COOLDOWN_SECONDS must be greater than zero.")
        fallback_model = os.getenv("FALLBACK_MODEL", "").strip() or None
        return cls(
            enabled=enabled,
            fallback_model=fallback_model,
            max_retries=max_retries,
            request_timeout_seconds=request_timeout,
            circuit_breaker_threshold=threshold,
            circuit_breaker_cooldown_seconds=cooldown,
        )


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_seconds: float) -> None:
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if not self._open_until:
                return True
            if time.monotonic() < self._open_until:
                return False
            self._open_until = 0.0
            self._failures = 0
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> bool:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and not self._open_until:
                self._open_until = time.monotonic() + self._cooldown_seconds
                return True
            return False


class ModelUnavailableError(RuntimeError):
    """Raised when the primary and configured fallback models both fail."""


def _invoke_with_timeout(function: Callable[[], T], timeout_seconds: float) -> T:
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result_queue.put((True, function()))
        except Exception as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=run, daemon=True, name="model-call")
    worker.start()
    try:
        succeeded, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"Model call exceeded {timeout_seconds:g} seconds."
        ) from exc
    if succeeded:
        return value
    raise value


def _retryable_error(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and 400 <= status_code < 500:
            return status_code == 429
        if isinstance(status_code, int) and status_code >= 500:
            return True
        if isinstance(current, (ValueError, TypeError, PermissionError)):
            return False
        if isinstance(current, (TimeoutError, ConnectionError, OSError)):
            return True
        if "timeout" in type(current).__name__.lower():
            return True
        if type(current).__name__.lower() in {
            "connecterror",
            "networkerror",
            "transporterror",
        }:
            return True
        if isinstance(current, RuntimeError):
            cause = current.__cause__ or current.__context__
            if cause is None:
                return True
            current = cause
            continue
        current = current.__cause__ or current.__context__
    return False


def _call_similarity(
    service: EmbeddingService,
    reference: str,
    candidate: str,
    user_id: str | None,
    timeout_seconds: float,
) -> float:
    return _invoke_with_timeout(
        lambda: service.similarity(reference, candidate, user_id=user_id),
        timeout_seconds,
    )


class ResilientEmbeddingService:
    def __init__(
        self,
        primary: EmbeddingService,
        fallback_factory: Callable[[], EmbeddingService] | None,
        settings: FallbackSettings,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._settings = settings
        self._sleep = sleep
        self._jitter = jitter
        self._primary_breaker = CircuitBreaker(
            settings.circuit_breaker_threshold,
            settings.circuit_breaker_cooldown_seconds,
        )
        self._fallback_breaker = CircuitBreaker(
            settings.circuit_breaker_threshold,
            settings.circuit_breaker_cooldown_seconds,
        )
        self._fallback_service: EmbeddingService | None = None
        self._fallback_lock = threading.Lock()
        self.model_name = getattr(primary, "model_name", primary.__class__.__qualname__)

    def similarity(
        self,
        reference: str,
        candidate: str,
        user_id: str | None = None,
    ) -> float:
        if not self._settings.enabled:
            return self._primary.similarity(reference, candidate, user_id=user_id)

        started = time.monotonic()
        primary_open = not self._primary_breaker.allow_request()
        if primary_open:
            logger.warning(
                "Fallback event dependency=primary_model outcome=circuit_open "
                "elapsed_seconds=%.3f",
                time.monotonic() - started,
            )
        else:
            last_error: Exception | None = None
            for attempt in range(self._settings.max_retries + 1):
                try:
                    result = _call_similarity(
                        self._primary,
                        reference,
                        candidate,
                        user_id,
                        self._settings.request_timeout_seconds,
                    )
                except Exception as exc:
                    last_error = exc
                    retryable = _retryable_error(exc)
                    opened = (
                        self._primary_breaker.record_failure()
                        if retryable
                        else False
                    )
                    if opened:
                        logger.warning(
                            "Fallback event dependency=primary_model error=%s "
                            "outcome=circuit_opened elapsed_seconds=%.3f",
                            exc,
                            time.monotonic() - started,
                        )
                        break
                    if not retryable or attempt >= self._settings.max_retries:
                        break
                    delay = self._jitter(
                        0.1,
                        min(2.0, 0.25 * (2**attempt)),
                    )
                    logger.warning(
                        "Fallback event dependency=primary_model error=%s "
                        "attempt=%s outcome=retry elapsed_seconds=%.3f",
                        exc,
                        attempt + 1,
                        time.monotonic() - started,
                    )
                    self._sleep(delay)
                else:
                    self._primary_breaker.record_success()
                    return result

            if last_error is not None and not _retryable_error(last_error):
                logger.error(
                    "Fallback event dependency=primary_model error=%s "
                    "outcome=not_retryable elapsed_seconds=%.3f",
                    last_error,
                    time.monotonic() - started,
                )
                raise last_error
            if last_error is not None:
                logger.warning(
                    "Fallback event dependency=primary_model error=%s "
                    "outcome=%s elapsed_seconds=%.3f",
                    last_error,
                    "retry_exhausted" if _retryable_error(last_error) else "not_retryable",
                    time.monotonic() - started,
                )

        if not self._settings.fallback_model or self._fallback_factory is None:
            logger.error(
                "Fallback event dependency=primary_model outcome=final_failure "
                "fallback_model=not_configured elapsed_seconds=%.3f",
                time.monotonic() - started,
            )
            raise ModelUnavailableError("The primary embedding model is unavailable.")
        if not self._fallback_breaker.allow_request():
            logger.error(
                "Fallback event dependency=fallback_model outcome=circuit_open "
                "elapsed_seconds=%.3f",
                time.monotonic() - started,
            )
            raise ModelUnavailableError("The configured fallback embedding model is unavailable.")

        try:
            fallback_service = self._get_fallback_service()
            result = _call_similarity(
                fallback_service,
                reference,
                candidate,
                user_id,
                self._settings.request_timeout_seconds,
            )
        except Exception as exc:
            opened = self._fallback_breaker.record_failure()
            logger.error(
                "Fallback event dependency=fallback_model error=%s "
                "outcome=failed circuit_opened=%s elapsed_seconds=%.3f",
                exc,
                opened,
                time.monotonic() - started,
            )
            raise ModelUnavailableError(
                "The primary and fallback embedding models are unavailable."
            ) from exc

        self._fallback_breaker.record_success()
        logger.warning(
            "Fallback event dependency=fallback_model outcome=used "
            "model=%s elapsed_seconds=%.3f",
            self._settings.fallback_model,
            time.monotonic() - started,
        )
        return result

    def _get_fallback_service(self) -> EmbeddingService:
        if self._fallback_service is not None:
            return self._fallback_service
        with self._fallback_lock:
            if self._fallback_service is None:
                self._fallback_service = _invoke_with_timeout(
                    self._fallback_factory,
                    self._settings.request_timeout_seconds,
                )
        return self._fallback_service
