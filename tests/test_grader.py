import unittest

from schemas import GradeRequest
from services.grader import Grader


class FixedEmbeddingService:
    def __init__(self, similarity: float) -> None:
        self._similarity = similarity

    def similarity(
        self, reference: str, candidate: str, user_id: str | None = None
    ) -> float:
        return self._similarity


class GraderTests(unittest.TestCase):
    def test_default_weights_prioritize_semantic_similarity(self) -> None:
        request = GradeRequest(
            question="ما وظيفة النبات؟",
            model_answer="يصنع النبات غذاءه",
            student_answer="النبات يصنع غذاءه",
            max_score=10,
            must_have_keywords=["النبات", "الغذاء"],
        )
        result = Grader(FixedEmbeddingService(0.6)).grade(request)

        self.assertEqual(request.weights.semantic, 0.7)
        self.assertEqual(request.weights.keywords, 0.2)
        self.assertEqual(request.weights.completeness, 0.1)
        expected_ratio = (
            result.semantic_similarity * 0.7
            + result.keyword_match_ratio * 0.2
            + result.completeness_ratio * 0.1
        )
        self.assertAlmostEqual(result.base_final_ratio, expected_ratio)

    def test_formula_rounding_and_bounds(self) -> None:
        grader = Grader(FixedEmbeddingService(0.8))
        request = GradeRequest(
            question="ما وظيفة النبات؟",
            model_answer="يصنع النبات غذاءه بعملية البناء الضوئي",
            student_answer="يصنع النبات غذاءه بالبناء الضوئي",
            max_score=10,
            must_have_keywords=["النبات", "البناء الضوئي"],
        )

        result = grader.grade(request)

        self.assertEqual(result.matched_keywords, ["النبات", "البناء الضوئي"])
        self.assertEqual(result.keyword_match_ratio, 1.0)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, result.max_score)
        self.assertEqual(result.final_ratio, result.base_final_ratio)

    def test_negation_mismatch_adds_review_flag_and_caps_score(self) -> None:
        grader = Grader(FixedEmbeddingService(0.98))
        request = GradeRequest(
            question="ما النتيجة؟",
            model_answer="الماء لا يغلي عند درجة الغرفة",
            student_answer="الماء يغلي عند درجة الغرفة",
            max_score=10,
        )

        result = grader.grade(request)

        self.assertTrue(result.contradiction_cap_applied)
        self.assertEqual(result.final_ratio, 0.5)
        self.assertGreater(result.base_final_ratio, result.final_ratio)
        self.assertTrue(any("النفي" in flag for flag in result.review_flags))

    def test_empty_keyword_list_does_not_divide_by_zero(self) -> None:
        grader = Grader(FixedEmbeddingService(0.5))
        request = GradeRequest(
            question="سؤال؟",
            model_answer="إجابة نموذجية",
            student_answer="إجابة",
            max_score=5,
            must_have_keywords=[],
        )

        result = grader.grade(request)

        self.assertEqual(result.keyword_match_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
