# Arabic Short-Answer Grader

A FastAPI service that supports teacher review of Arabic short-answer and descriptive responses. It combines multilingual semantic similarity, teacher-provided or optionally extracted keywords, and a bounded answer-length completeness signal. It is decision support, not a guarantee of accurate grading.

## Project files

- `main.py` — FastAPI application, startup model initialization, `/health`, and `/grade`.
- `schemas.py` — validated Pydantic request, grading-weight, and response models.
- `services/embedding_protocol.py` — provider interface for normalized similarity scores.
- `services/embeddings.py` — local SentenceTransformers implementation of the provider interface.
- `services/grader.py` — hybrid score calculation, negation review checks, and Arabic feedback.
- `utils/nlp.py` — Arabic normalization, tokenization, keyword extraction, and matching.
- `tests/test_nlp.py` — tests for normalization and keyword utility behavior.
- `tests/test_grader.py` — tests for score bounds, formula behavior, and negation handling.
- `requirements.txt` — runtime and test dependencies.
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

The local model is loaded once during application startup. A load or inference error is surfaced as a startup failure or HTTP `503`, respectively. `GET /health` returns service readiness.

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
    "semantic": 0.5,
    "keywords": 0.4,
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

The default base ratio is:

`(semantic_similarity × 0.5) + (keyword_match_ratio × 0.4) + (completeness_ratio × 0.1)`

The result is multiplied by `max_score`, rounded to the requested precision, and bounded to `[0, max_score]`. SentenceTransformer cosine similarity is mapped from `[-1, 1]` to `[0, 1]`; it is one signal, not proof of correctness. Keyword matching uses normalized whole terms/phrases, with optional fuzzy matching. Even an exact keyword match cannot establish whether a student used the concept correctly.

Completeness is a simple word-count ratio: it reaches `1.0` at 70% of the model answer's normalized token count and contributes at most 10% under the default weights. It limits length-only rewards for verbose answers but cannot distinguish relevant explanation from filler or guarantee that a concise response is complete.

The contradiction check conservatively flags a possible mismatch when one answer contains a recognized negation marker, the other does not, and they share at least two normalized terms. It does not assert that the student is wrong. When triggered, it adds a review flag and caps the final ratio at `0.5`; `base_final_ratio` preserves the uncapped weighted calculation, and `contradiction_cap_applied` makes the adjustment explicit. This rule is intentionally simple and can miss or flag context-dependent negation; teachers should review flagged answers.

Student answers are not logged by the application. For educational use, teachers should inspect the returned score, breakdown, and review flags rather than treating automatic grades as definitive.

## Tests

```powershell
python -m unittest discover -s tests -v
```
