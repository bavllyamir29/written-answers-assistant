# Arabic Short-Answer Grader

A FastAPI service that supports teacher review of Arabic short-answer and descriptive responses. It combines multilingual semantic similarity, teacher-provided or optionally extracted keywords, and a bounded answer-length completeness signal. It is decision support, not a guarantee of accurate grading.

## Project files

- `main.py` — FastAPI application, startup model initialization, `/health`, and `/grade`.
- `schemas.py` — validated Pydantic request, grading-weight, and response models.
- `services/embedding_protocol.py` — provider interface for normalized similarity scores.
- `services/embeddings.py` — local SentenceTransformers implementation of the provider interface.
- `services/embedding_cache.py` — optional Redis cache wrapper for the local semantic-similarity result.
- `services/resilience.py` — configurable retries, model fallback, request timeouts, and circuit breakers.
- `services/grader.py` — hybrid score calculation, negation review checks, and Arabic feedback.
- `logger.py` — grading decorator, JSON rotating file logging, SQLite audit tables, and a teacher-override logging hook.
- `utils/nlp.py` — Arabic normalization, tokenization, keyword extraction, and matching.
- `tests/test_nlp.py` — tests for normalization and keyword utility behavior.
- `tests/test_grader.py` — tests for score bounds, formula behavior, and negation handling.
- `requirements.txt` — runtime and test dependencies.
- `requirements-cache.txt` — optional Redis client dependency for the cache.
- `.env.example` — optional embedding model and device configuration.

## Setup and run

Python 3.10 or newer is required. The first startup downloads the configured SentenceTransformers model if it is not already cached; allow sufficient disk space and network access.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload
```

The application loads an optional `.env` file at startup. `EMBEDDING_MODEL` and `EMBEDDING_DEVICE` are optional. The default model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. No OpenAI key or remote embedding provider is required.

The primary local model is loaded once during application startup. If it cannot load and graceful degradation is enabled, the configured fallback model is tried; if neither model can load, the API starts in a degraded state and grading requests receive the documented unavailable response. `GET /health` reports readiness.

## Optional Redis cache

This project has no LLM/chat-completion API call. Its only model inference is the local embedding operation used to calculate semantic similarity, so this cache stores the resulting similarity number—not an LLM response or the complete grade. It does not change the model name, the `.encode()` call, or the API response schema. The embedding inputs are passed to the model unchanged; Arabic normalization is only used to derive the cache key. The key includes the model name, both input texts, embedding/scoring parameters, and `student_id` (if provided), is SHA-256 hashed, and uses the `llm:v1:` prefix. Bump the prefix version in `services/embedding_cache.py` to invalidate existing entries after changing key or similarity behavior.

Install the Redis client with `pip install -r requirements-cache.txt`, and run a Redis server. The optional settings in `.env.example` default to enabled caching, a 3600-second TTL, and `redis://localhost:6379/0`. If Redis is unavailable or times out, the app logs a warning, records an INFO-level hit/miss, and computes similarity normally. Redis connection and socket timeouts are 150 ms. To turn caching off without changing code, set `CACHE_ENABLED=false`; to remove the wrapper entirely, remove its single construction line in `main.py`.

Similarity requests in this project are independent (there is no conversation history, tool/function calling, or streaming), so those LLM-specific cache exclusions do not arise. If a later model interface adds tools or context, include every output-affecting input in the key or bypass caching for that request. Cache hit/miss messages are emitted at INFO through Uvicorn's `uvicorn.error` logger.

## Graceful degradation

The project performs local SentenceTransformers inference; it does not call an LLM API. With `FALLBACK_ENABLED=true`, each local model inference has a `REQUEST_TIMEOUT_SECONDS` wait limit. Transient errors (timeouts, connection errors, server-side status codes, and rate limits if a provider later exposes them) are retried up to `MAX_RETRIES` times (configurable from 0 to 2), with exponential backoff and jitter. Bad-request/authentication 4xx errors other than 429 and invalid input errors are not retried. After primary inference failures, the service attempts the optional `FALLBACK_MODEL`, which must be a SentenceTransformers-compatible model identifier. The existing primary model and encode parameters are unchanged on the normal path.

Model, Redis, and SQLite audit-database dependencies use circuit breakers controlled by `CIRCUIT_BREAKER_THRESHOLD` and `CIRCUIT_BREAKER_COOLDOWN_SECONDS`. Redis reads/writes remain fail-open and use 150 ms socket timeouts. Cache, model, and audit-database fallback events are written to the application logger with the dependency, error/outcome, and elapsed time where applicable.

If both models are unavailable, including at startup, the API stays available and returns the normal `GradeResponse` fields with a zero placeholder score, zero component ratios, a localized temporary-unavailability message, and a review flag explicitly warning not to use that score. `/health` reports `degraded` if no model could be loaded. This preserves response shape; it does not represent an earned grade. When `FALLBACK_ENABLED=false`, retries/model fallback/final placeholder handling are bypassed and the existing startup/HTTP 503 behavior is retained. Redis remains independently configurable with `CACHE_ENABLED`.

Python cannot forcibly stop a local Torch/SentenceTransformers inference already executing in a worker thread. A timed-out request proceeds to retry/fallback, but that underlying computation may continue briefly in the background. Configure a practical timeout and monitor CPU/GPU capacity.

## Audit logging

`Grader.grade` is wrapped with `@log_grading`; its grading calculation and response object are otherwise unchanged. Each completed or failed call is logged to `logs/app.log` as UTF-8 JSON Lines with Arabic preserved (`ensure_ascii=False`). The file rotates at 5 MB and keeps 5 backups. SQLite tables are created automatically in `logs/grading.db` (`GRADING_LOG_DB_PATH` can override the database location).

The `grading_logs` table records a unique request ID, timestamp, hashed student ID, optional question ID, model/student answers, score, Arabic feedback, embedding model name, token/cost fields, elapsed seconds, status, and error/traceback on failure. The local embedding service does not expose token usage or per-request cost, so those fields are `null`. This project has no existing teacher override endpoint or score-edit flow; `log_teacher_override(...)` in `logger.py` is provided for a future existing override handler to call, and writes to both the JSON log and `teacher_overrides` table. Supply the grading `request_id` when calling it to relate an override to its grading record.

Pass `student_id` and, if available, `question_id` in the grading request for those fields to be populated. Student IDs are SHA-256 hashed by default; setting `STUDENT_ID_HASH_SALT` uses keyed HMAC-SHA-256 and is recommended for production. Student answers and model answers are deliberately included in both requested audit destinations and therefore stored as plaintext. Restrict filesystem/database access and define appropriate retention policies for student data. Audit database/file errors are logged; database logging has a circuit breaker and does not prevent grading from returning a result.

## API

### `POST /grade`

Example request:

```json
{
  "question": "ما أهمية البناء الضوئي للنبات؟",
  "model_answer": "تساعد عملية البناء الضوئي النبات على صنع غذائه باستخدام ضوء الشمس.",
  "student_answer": "يصنع النبات غذاءه بعملية البناء الضوئي وبوجود ضوء الشمس.",
  "max_score": 5,
  "must_have_keywords": ["البناء الضوئي", "ضوء الشمس"],
  "allow_fuzzy_matching": true,
  "fuzzy_threshold": 0.88,
  "weights": {
    "semantic": 0.7,
    "keywords": 0.2,
    "completeness": 0.1
  },
  "score_decimals": 2
}
```

Weights must be in `[0, 1]` and sum to `1`. `max_score` must be greater than zero. `score_decimals` ranges from 0 to 4. `normalize_ta_marbuta` defaults to `false`, since changing ة to ه can collapse distinct spellings and affect meaning.

If `must_have_keywords` is omitted or empty and `extract_keywords` is `true`, the service selects up to eight frequent non-stopword tokens from the model answer. These automatically selected terms can be poor proxies for scientific or subject-specific concepts and should be reviewed by the teacher. If there are no keywords to score, `keyword_match_ratio` is defined as `1.0`; there is no keyword-based deduction.

Example response shape:

```json
{
  "score": 4.65,
  "max_score": 5.0,
  "final_ratio": 0.93,
  "base_final_ratio": 0.93,
  "semantic_similarity": 0.9,
  "keyword_match_ratio": 1.0,
  "completeness_ratio": 1.0,
  "matched_keywords": ["البناء الضوئي", "ضوء الشمس"],
  "missing_keywords": [],
  "feedback_ar": "ظهرت في الإجابة مصطلحات مطلوبة مثل: البناء الضوئي، ضوء الشمس. يوجد تقارب دلالي جيد مع الإجابة النموذجية.",
  "review_flags": [],
  "contradiction_cap_applied": false
}
```

Feedback is generated from the computed similarity, term matching, answer length, and review checks. Keyword presence indicates a textual match only; it does not prove that a term was used correctly.
The required `question` field is retained as request context; the current grading formula compares the model answer directly with the student answer.

### cURL

```powershell
curl.exe -X POST "http://127.0.0.1:8000/grade" `
  -H "Content-Type: application/json" `
  --data-raw "{\"question\":\"ما وظيفة الجذور؟\",\"model_answer\":\"تمتص الجذور الماء والأملاح من التربة.\",\"student_answer\":\"تمتص الجذور الماء والأملاح من التربة.\",\"max_score\":4,\"must_have_keywords\":[\"الماء\",\"الأملاح\"]}"
```

### Python

```python
import requests

payload = {
    "question": "لماذا تحتاج النباتات إلى ضوء الشمس؟",
    "model_answer": "تحتاج النباتات إلى ضوء الشمس لإتمام عملية البناء الضوئي وصنع الغذاء.",
    "student_answer": "يستخدم النبات ضوء الشمس في البناء الضوئي لصنع غذائه.",
    "max_score": 5,
    "must_have_keywords": ["ضوء الشمس", "البناء الضوئي"],
}

response = requests.post("http://127.0.0.1:8000/grade", json=payload, timeout=120)
response.raise_for_status()
print(response.json())
```

Install the client for the Python example with `pip install requests`.

## Scoring and limitations

The default base ratio gives more influence to meaning than exact term coverage:

`(semantic_similarity × 0.7) + (keyword_match_ratio × 0.2) + (completeness_ratio × 0.1)`

The result is multiplied by `max_score`, rounded to the requested precision, and bounded to `[0, max_score]`. You can still supply custom weights, provided they sum to `1`. SentenceTransformer cosine similarity is mapped from `[-1, 1]` to `[0, 1]`; it is one signal, not proof of correctness. Keyword matching uses normalized whole terms/phrases, with optional fuzzy matching. Even an exact keyword match cannot establish whether a student used the concept correctly.

Completeness is a simple word-count ratio: it reaches `1.0` at 70% of the model answer's normalized token count and contributes at most 10% under the default weights. It limits length-only rewards for verbose answers but cannot distinguish relevant explanation from filler or guarantee that a concise response is complete.

The contradiction check conservatively flags a possible mismatch when one answer contains a recognized negation marker, the other does not, and they share at least two normalized terms. It does not assert that the student is wrong. When triggered, it adds a review flag and caps the final ratio at `0.5`; `base_final_ratio` preserves the uncapped weighted calculation, and `contradiction_cap_applied` makes the adjustment explicit. This rule is intentionally simple and can miss or flag context-dependent negation; teachers should review flagged answers.

Student answers are not logged by the application. For educational use, teachers should inspect the returned score, breakdown, and review flags rather than treating automatic grades as definitive.

## Tests

```powershell
python -m unittest discover -s tests -v
```
