import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670]")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_NEGATION_WORDS = frozenset({"لا", "ليس", "ليست", "لم", "لن", "ما", "غير", "عدم"})
NEGATION_WORDS = _NEGATION_WORDS

_ARABIC_STOPWORDS = frozenset(
    {
        "التي", "الذي", "الذين", "اللاتي", "هذا", "هذه", "ذلك", "تلك",
        "هو", "هي", "هم", "هن", "في", "من", "على", "عن", "إلى", "الى",
        "مع", "بين", "بعد", "قبل", "عند", "كان", "كانت", "يكون", "تكون",
        "أن", "ان", "إن", "اذا", "إذا", "ثم", "كما", "و", "أو", "او",
        "ف", "ب", "ل", "ال", "كل", "قد", "تم", "لأن", "لان",
    }
)


def normalize_arabic(text: str, normalize_ta_marbuta: bool = False) -> str:
    """Normalize Arabic orthography while preserving negation and word boundaries."""
    text = _TASHKEEL.sub("", text.replace("ـ", ""))
    text = text.translate(_ALEF_VARIANTS).replace("ى", "ي")
    if normalize_ta_marbuta:
        text = text.replace("ة", "ه")

    normalized_chars: list[str] = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            normalized_chars.append(" ")
        else:
            normalized_chars.append(char)
    return _WHITESPACE.sub(" ", "".join(normalized_chars)).strip()


def tokenize(text: str, normalize_ta_marbuta: bool = False) -> list[str]:
    normalized = normalize_arabic(text, normalize_ta_marbuta)
    return _TOKEN.findall(normalized)


def extract_candidate_terms(
    model_answer: str, normalize_ta_marbuta: bool = False, limit: int = 8
) -> list[str]:
    """Return frequent non-stopword tokens as reviewable candidate keywords."""
    tokens = [
        token
        for token in tokenize(model_answer, normalize_ta_marbuta)
        if len(token) > 2 and token not in _ARABIC_STOPWORDS
    ]
    counts = Counter(tokens)
    first_seen = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token]))
    return ranked[:limit]


def match_keywords(
    keywords: list[str],
    student_answer: str,
    allow_fuzzy: bool = True,
    threshold: float = 0.88,
    normalize_ta_marbuta: bool = False,
) -> tuple[list[str], list[str]]:
    """Match whole normalized terms, optionally allowing fuzzy token/phrase matches."""
    normalized_answer = normalize_arabic(student_answer, normalize_ta_marbuta)
    answer_tokens = normalized_answer.split()
    matched: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for keyword in keywords:
        normalized_keyword = normalize_arabic(keyword, normalize_ta_marbuta)
        if not normalized_keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)
        if _contains_phrase(normalized_answer, normalized_keyword):
            matched.append(keyword)
        elif allow_fuzzy and _fuzzy_contains(
            normalized_keyword.split(), answer_tokens, threshold
        ):
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _fuzzy_contains(keyword_tokens: list[str], answer_tokens: list[str], threshold: float) -> bool:
    if not keyword_tokens or len(answer_tokens) < len(keyword_tokens):
        return False
    width = len(keyword_tokens)
    for start in range(len(answer_tokens) - width + 1):
        candidate = " ".join(answer_tokens[start : start + width])
        target = " ".join(keyword_tokens)
        if SequenceMatcher(None, target, candidate).ratio() >= threshold:
            return True
    return False
