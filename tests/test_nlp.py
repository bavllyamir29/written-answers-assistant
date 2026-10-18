import unittest

from utils.nlp import extract_candidate_terms, match_keywords, normalize_arabic


class ArabicTextTests(unittest.TestCase):
    def test_removes_diacritics_and_tatweel_and_normalizes_alif_and_ya(self) -> None:
        self.assertEqual(
            normalize_arabic("إِنَّ الآثارَ عَلَى الفَتَى"),
            "ان الاثار علي الفتي",
        )

    def test_preserves_negation(self) -> None:
        self.assertEqual(normalize_arabic("لا يذوب، ولم يتغير"), "لا يذوب ولم يتغير")

    def test_ta_marbuta_normalization_is_opt_in(self) -> None:
        self.assertEqual(normalize_arabic("مدرسة"), "مدرسة")
        self.assertEqual(normalize_arabic("مدرسة", normalize_ta_marbuta=True), "مدرسه")

    def test_exact_matching_uses_whole_terms_and_fuzzy_is_configurable(self) -> None:
        matched, missing = match_keywords(
            ["الخلية", "الانقسام"], "تشرح الخلايا الانقسام", allow_fuzzy=False
        )
        self.assertEqual(matched, ["الانقسام"])
        self.assertEqual(missing, ["الخلية"])

        matched, missing = match_keywords(
            ["الخلية"], "الخلية", allow_fuzzy=True, threshold=0.8
        )
        self.assertEqual(matched, ["الخلية"])
        self.assertEqual(missing, [])

    def test_candidate_term_extraction_omits_common_words(self) -> None:
        terms = extract_candidate_terms("تحدث عملية البناء الضوئي في النبات")
        self.assertIn("البناء", terms)
        self.assertNotIn("في", terms)


if __name__ == "__main__":
    unittest.main()
