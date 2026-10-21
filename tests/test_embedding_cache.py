import json
import unittest

from services.embedding_cache import (
    CACHE_KEY_PREFIX,
    RedisCachedEmbeddingService,
    cache_key,
    normalize_cache_text,
)


class FakeEmbeddingService:
    model_name = "fixture-model"

    def __init__(self, value: float = 0.75, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        self.calls.append((reference, candidate))
        if self.error:
            raise self.error
        return self.value


class FakeRedis:
    def __init__(self, fail_get: bool = False, fail_set: bool = False) -> None:
        self.values: dict[str, str] = {}
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.ttls: dict[str, int] = {}
        self.get_calls = 0

    def get(self, key: str) -> str | None:
        self.get_calls += 1
        if self.fail_get:
            raise TimeoutError("Redis read timed out")
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int) -> bool:
        if self.fail_set:
            raise TimeoutError("Redis write timed out")
        self.values[key] = value
        self.ttls[key] = ex
        return True


class EmbeddingCacheTests(unittest.TestCase):
    def test_normalization_is_for_key_content(self) -> None:
        self.assertEqual(
            normalize_cache_text("  إِنَّـــهُ  TEST   ١۲ "),
            "انه test 12",
        )

    def test_key_is_stable_scoped_and_versioned(self) -> None:
        first = cache_key(
            model_name="model",
            reference="إجابة ١",
            candidate="  TEST ",
            user_id="student-a",
        )
        equivalent = cache_key(
            model_name="model",
            reference="اجابة 1",
            candidate="test",
            user_id="student-a",
        )
        other_user = cache_key(
            model_name="model",
            reference="اجابة 1",
            candidate="test",
            user_id="student-b",
        )

        self.assertTrue(first.startswith(CACHE_KEY_PREFIX))
        self.assertEqual(first, equivalent)
        self.assertNotEqual(first, other_user)

    def test_miss_caches_and_hit_returns_same_float_without_model_call(self) -> None:
        embeddings = FakeEmbeddingService()
        redis = FakeRedis()
        cached = RedisCachedEmbeddingService(embeddings, redis, ttl_seconds=75)

        with self.assertLogs("uvicorn.error", level="INFO") as log_output:
            first = cached.similarity("إجابةـ أصلية", "ردّ", user_id="student-1")
            second = cached.similarity("اجابة اصلية", "رد", user_id="student-1")

        self.assertEqual(first, 0.75)
        self.assertEqual(second, first)
        self.assertEqual(len(embeddings.calls), 1)
        self.assertEqual(embeddings.calls[0], ("إجابةـ أصلية", "ردّ"))
        self.assertEqual(list(redis.ttls.values()), [75])
        self.assertEqual(json.loads(next(iter(redis.values.values()))), first)
        self.assertTrue(any("cache miss" in message for message in log_output.output))
        self.assertTrue(any("cache hit" in message for message in log_output.output))

    def test_cache_key_separates_users(self) -> None:
        embeddings = FakeEmbeddingService()
        cached = RedisCachedEmbeddingService(embeddings, FakeRedis(), ttl_seconds=60)

        cached.similarity("answer", "candidate", user_id="one")
        cached.similarity("answer", "candidate", user_id="two")

        self.assertEqual(len(embeddings.calls), 2)

    def test_invalid_cached_value_is_ignored_and_replaced(self) -> None:
        embeddings = FakeEmbeddingService(value=0.3)
        redis = FakeRedis()
        key = cache_key(
            model_name=embeddings.model_name,
            reference="ref",
            candidate="candidate",
        )
        redis.values[key] = ""
        cached = RedisCachedEmbeddingService(embeddings, redis, ttl_seconds=60)

        self.assertEqual(cached.similarity("ref", "candidate"), 0.3)
        self.assertEqual(len(embeddings.calls), 1)
        self.assertEqual(json.loads(redis.values[key]), 0.3)

    def test_redis_read_and_write_failures_do_not_block_model_result(self) -> None:
        for redis in (FakeRedis(fail_get=True), FakeRedis(fail_set=True)):
            with self.subTest(redis=redis):
                embeddings = FakeEmbeddingService(value=0.6)
                cached = RedisCachedEmbeddingService(embeddings, redis, ttl_seconds=60)
                self.assertEqual(cached.similarity("ref", "candidate"), 0.6)
                self.assertEqual(len(embeddings.calls), 1)

    def test_redis_circuit_breaker_skips_reads_when_open(self) -> None:
        embeddings = FakeEmbeddingService(value=0.6)
        redis = FakeRedis(fail_get=True)
        cached = RedisCachedEmbeddingService(
            embeddings,
            redis,
            ttl_seconds=60,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown_seconds=30,
        )

        for _ in range(3):
            self.assertEqual(cached.similarity("ref", "candidate"), 0.6)

        self.assertEqual(redis.get_calls, 2)
        self.assertEqual(len(embeddings.calls), 3)

    def test_disabled_cache_bypasses_redis(self) -> None:
        embeddings = FakeEmbeddingService(value=0.4)
        cached = RedisCachedEmbeddingService(
            embeddings, redis_client=None, ttl_seconds=60, enabled=False
        )

        self.assertEqual(cached.similarity("ref", "candidate"), 0.4)
        self.assertEqual(embeddings.calls, [("ref", "candidate")])

    def test_failed_model_call_is_not_cached(self) -> None:
        redis = FakeRedis()
        embeddings = FakeEmbeddingService(error=RuntimeError("inference failed"))
        cached = RedisCachedEmbeddingService(embeddings, redis, ttl_seconds=60)

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            cached.similarity("ref", "candidate")

        self.assertEqual(redis.values, {})


if __name__ == "__main__":
    unittest.main()
