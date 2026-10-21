import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Protocol

from services.embedding_protocol import EmbeddingService
from services.resilience import CircuitBreaker

logger = logging.getLogger("uvicorn.error")

CACHE_KEY_PREFIX = "llm:v1:"
_TASHKEEL = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_WHITESPACE = re.compile(r"\s+")
_ALEF_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        **{chr(0x0660 + digit): str(digit) for digit in range(10)},
        **{chr(0x06F0 + digit): str(digit) for digit in range(10)},
    }
)


class RedisClient(Protocol):
    def get(self, key: str) -> str | bytes | None: ...

    def set(self, key: str, value: str, ex: int) -> Any: ...


def normalize_cache_text(text: str) -> str:
    """Normalize text only for cache-key construction; never alter model inputs."""
    text = _TASHKEEL.sub("", text.replace("ـ", ""))
    text = text.translate(_ALEF_TRANSLATION).lower()
    return _WHITESPACE.sub(" ", text).strip()


def _is_similarity_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric_value = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(numeric_value) and 0.0 <= numeric_value <= 1.0


def cache_key(
    *,
    model_name: str,
    reference: str,
    candidate: str,
    user_id: str | None = None,
    generation_parameters: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model_name": model_name,
        "system_prompt": None,
        "messages": [
            {"role": "reference", "content": normalize_cache_text(reference)},
            {"role": "candidate", "content": normalize_cache_text(candidate)},
        ],
        "generation_parameters": generation_parameters
        if generation_parameters is not None
        else {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "similarity_mapping": {"low": 0.45, "high": 0.85},
        },
        "user_id": user_id,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


class RedisCachedEmbeddingService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        redis_client: RedisClient | None,
        ttl_seconds: int,
        enabled: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 30,
        fallback_enabled: bool = True,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("CACHE_TTL_SECONDS must be greater than zero.")
        self._embedding_service = embedding_service
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled
        self._breaker_enabled = enabled and fallback_enabled
        self._breaker = CircuitBreaker(
            circuit_breaker_threshold, circuit_breaker_cooldown_seconds
        )
        self.model_name = getattr(
            embedding_service,
            "model_name",
            embedding_service.__class__.__qualname__,
        )

    @classmethod
    def from_environment(
        cls,
        embedding_service: EmbeddingService,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 30,
        fallback_enabled: bool = True,
    ) -> "RedisCachedEmbeddingService":
        enabled = os.getenv("CACHE_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        redis_client: RedisClient | None = None
        if enabled:
            try:
                import redis

                redis_client = redis.Redis.from_url(
                    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    socket_connect_timeout=0.15,
                    socket_timeout=0.15,
                    retry_on_timeout=False,
                    decode_responses=True,
                )
            except Exception as exc:
                logger.warning(
                    "Fallback event dependency=redis_cache error=%s "
                    "outcome=continue_without_cache",
                    exc,
                )
        return cls(
            embedding_service,
            redis_client,
            ttl_seconds,
            enabled,
            circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds,
            fallback_enabled,
        )

    def similarity(
        self,
        reference: str,
        candidate: str,
        user_id: str | None = None,
    ) -> float:
        if not self._enabled:
            logger.info("Embedding cache miss reason=disabled")
            return self._embedding_service.similarity(
                reference, candidate, user_id=user_id
            )

        key = cache_key(
            model_name=self.model_name,
            reference=reference,
            candidate=candidate,
            user_id=user_id,
        )
        if self._redis is None:
            logger.warning(
                "Fallback event dependency=redis_cache error=client_unavailable "
                "outcome=continue_without_cache"
            )
            logger.info("Embedding cache miss key=%s", key)
            return self._embedding_service.similarity(reference, candidate)

        if self._breaker_enabled and not self._breaker.allow_request():
            logger.warning("Fallback event dependency=redis_cache outcome=circuit_open")
            logger.info("Embedding cache miss key=%s", key)
            return self._embedding_service.similarity(
                reference, candidate, user_id=user_id
            )

        cache_started = time.monotonic()
        try:
            cached_value = self._redis.get(key)
        except Exception as exc:
            opened = self._breaker.record_failure() if self._breaker_enabled else False
            logger.warning("Redis cache read failed; continuing without cache: %s", exc)
            logger.info("Embedding cache miss key=%s", key)
            logger.warning(
                "Fallback event dependency=redis_cache error=%s outcome=skip_cache "
                "circuit_opened=%s elapsed_seconds=%.3f",
                exc,
                opened,
                time.monotonic() - cache_started,
            )
            return self._embedding_service.similarity(
                reference, candidate, user_id=user_id
            )
        if self._breaker_enabled:
            self._breaker.record_success()

        if cached_value is not None:
            try:
                value = json.loads(
                    cached_value.decode("utf-8")
                    if isinstance(cached_value, bytes)
                    else cached_value
                )
                if _is_similarity_value(value):
                    logger.info("Embedding cache hit key=%s", key)
                    return float(value)
                logger.warning("Ignoring invalid Redis cache value for key=%s", key)
            except Exception as exc:
                logger.warning("Ignoring invalid Redis cache value for key=%s: %s", key, exc)

        logger.info("Embedding cache miss key=%s", key)
        result = self._embedding_service.similarity(
            reference, candidate, user_id=user_id
        )
        if _is_similarity_value(result):
            cache_started = time.monotonic()
            try:
                self._redis.set(
                    key,
                    json.dumps(float(result), allow_nan=False),
                    ex=self._ttl_seconds,
                )
            except Exception as exc:
                opened = self._breaker.record_failure() if self._breaker_enabled else False
                logger.warning("Redis cache write failed; returning model result: %s", exc)
                logger.warning(
                    "Fallback event dependency=redis_cache error=%s "
                    "outcome=return_model_result circuit_opened=%s "
                    "elapsed_seconds=%.3f",
                    exc,
                    opened,
                    time.monotonic() - cache_started,
                )
            else:
                if self._breaker_enabled:
                    self._breaker.record_success()
        return result
