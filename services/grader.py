from schemas import GradeRequest, GradeResponse
from services.embedding_protocol import EmbeddingService
from utils.nlp import (
    NEGATION_WORDS,
    extract_candidate_terms,
    match_keywords,
    normalize_arabic,
    tokenize,
)

CONTRADICTION_SHARED_TERM_MINIMUM = 2
CONTRADICTION_RATIO_CAP = 0.5


class Grader:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service

    def grade(self, request: GradeRequest) -> GradeResponse:
        normalize_ta = request.normalize_ta_marbuta
        semantic_similarity = self._embedding_service.similarity(
            request.model_answer, request.student_answer
        )

        keywords = request.must_have_keywords or []
        automatically_extracted = False
        if not keywords and request.extract_keywords:
            keywords = extract_candidate_terms(
                request.model_answer,
                normalize_ta_marbuta=normalize_ta,
            )
            automatically_extracted = True

        matched, missing = match_keywords(
            keywords,
            request.student_answer,
            allow_fuzzy=request.allow_fuzzy_matching,
            threshold=request.fuzzy_threshold,
            normalize_ta_marbuta=normalize_ta,
        )
        keyword_match_ratio = len(matched) / len(keywords) if keywords else 1.0
        completeness_ratio = self._completeness_ratio(
            request.model_answer, request.student_answer, normalize_ta
        )

        weights = request.weights
        base_final_ratio = (
            semantic_similarity * weights.semantic
            + keyword_match_ratio * weights.keywords
            + completeness_ratio * weights.completeness
        )
        base_final_ratio = min(1.0, max(0.0, base_final_ratio))

        review_flags: list[str] = []
        if automatically_extracted:
            review_flags.append(
                "تم استخراج المصطلحات تلقائيًا؛ يُنصح بمراجعتها قبل اعتماد الدرجة."
            )

        possible_negation_mismatch = self._has_possible_negation_mismatch(
            request.model_answer,
            request.student_answer,
            normalize_ta,
        )
        final_ratio = base_final_ratio
        contradiction_cap_applied = False
        if possible_negation_mismatch:
            review_flags.append(
                "قد يوجد اختلاف في النفي بين الإجابتين؛ يلزم فحص الإجابة يدويًا."
            )
            if final_ratio > CONTRADICTION_RATIO_CAP:
                final_ratio = CONTRADICTION_RATIO_CAP
                contradiction_cap_applied = True

        score = round(final_ratio * request.max_score, request.score_decimals)
        score = min(request.max_score, max(0.0, score))
        feedback = self._feedback(
            semantic_similarity,
            completeness_ratio,
            matched,
            missing,
            possible_negation_mismatch,
        )

        return GradeResponse(
            score=score,
            max_score=request.max_score,
            final_ratio=final_ratio,
            base_final_ratio=base_final_ratio,
            semantic_similarity=semantic_similarity,
            keyword_match_ratio=keyword_match_ratio,
            completeness_ratio=completeness_ratio,
            matched_keywords=matched,
            missing_keywords=missing,
            feedback_ar=feedback,
            review_flags=review_flags,
            contradiction_cap_applied=contradiction_cap_applied,
        )

    @staticmethod
    def _completeness_ratio(
        model_answer: str, student_answer: str, normalize_ta_marbuta: bool
    ) -> float:
        expected_words = len(tokenize(model_answer, normalize_ta_marbuta))
        student_words = len(tokenize(student_answer, normalize_ta_marbuta))
        if student_words == 0:
            return 0.0
        # Length stops contributing after the student reaches 70% of the model answer.
        target_words = max(1.0, expected_words * 0.7)
        return min(1.0, student_words / target_words)

    @staticmethod
    def _has_possible_negation_mismatch(
        model_answer: str, student_answer: str, normalize_ta_marbuta: bool
    ) -> bool:
        model_tokens = set(tokenize(model_answer, normalize_ta_marbuta))
        student_tokens = set(tokenize(student_answer, normalize_ta_marbuta))
        model_is_negated = bool(model_tokens & NEGATION_WORDS)
        student_is_negated = bool(student_tokens & NEGATION_WORDS)
        shared_content_terms = model_tokens & student_tokens - NEGATION_WORDS
        return (
            model_is_negated != student_is_negated
            and len(shared_content_terms) >= CONTRADICTION_SHARED_TERM_MINIMUM
        )

    @staticmethod
    def _feedback(
        semantic_similarity: float,
        completeness_ratio: float,
        matched: list[str],
        missing: list[str],
        possible_negation_mismatch: bool,
    ) -> str:
        parts: list[str] = []
        if matched:
            parts.append("ظهرت في الإجابة مصطلحات مطلوبة مثل: " + "، ".join(matched) + ".")
        if missing:
            parts.append("لم تظهر المصطلحات المطلوبة التالية: " + "، ".join(missing) + ".")
        if semantic_similarity >= 0.7:
            parts.append("يوجد تقارب دلالي جيد مع الإجابة النموذجية.")
        elif semantic_similarity < 0.4:
            parts.append("التقارب الدلالي مع الإجابة النموذجية محدود.")
        else:
            parts.append("يوجد تقارب دلالي جزئي مع الإجابة النموذجية.")
        if completeness_ratio < 0.5:
            parts.append("الإجابة قصيرة مقارنة بالإجابة النموذجية.")
        if possible_negation_mismatch:
            parts.append("قد يغيّر اختلاف أداة النفي المعنى؛ يرجى مراجعة الإجابة يدويًا.")
        if not parts:
            parts.append("لم تُحدّد مصطلحات إلزامية؛ تعتمد الدرجة على التقارب الدلالي وطول الإجابة.")
        return " ".join(parts)
