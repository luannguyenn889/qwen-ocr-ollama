import unittest
from app.core.vietnamese_spell_corrector import correct_vietnamese_spelling


class VietnameseSpellCorrectorTests(unittest.TestCase):
    def test_correct_isolated_non_words(self):
        # Non-words (nghỉên, chuẫn) should be corrected to (nghiên, chuẩn)
        wrong = "Tài liệu này nghỉên cứu về chuẫn mực."
        expected = "Tài liệu này nghiên cứu về chuẩn mực."
        corrected, count = correct_vietnamese_spelling(wrong)
        self.assertEqual(corrected, expected)
        self.assertGreaterEqual(count, 1)

    def test_correct_trigrams_and_bigrams(self):
        text = "Uỷ ban nhân dân thành phố ban hành quyết định."
        corrected, _ = correct_vietnamese_spelling(text)
        self.assertEqual(corrected, text)

    def test_preserve_acronyms_and_english(self):
        text = "Hệ thống sử dụng mô hình AI và chuẩn REST API."
        corrected, _ = correct_vietnamese_spelling(text)
        self.assertEqual(corrected, text)

    def test_fuzzy_bigram_correction(self):
        # bộc lập -> độc lập (bộc is 1 char edit from độc, độc lập is in bigrams)
        wrong = "Quyền bộc lập và tự do của dân tộc."
        expected = "Quyền độc lập và tự do của dân tộc."
        corrected, count = correct_vietnamese_spelling(wrong)
        self.assertEqual(corrected, expected)
        self.assertGreaterEqual(count, 1)

        # The giờn -> Thế gian / Thế giới
        wrong_phrase = "The giờn ngày nay đã phát triển."
        corrected_phrase, _ = correct_vietnamese_spelling(wrong_phrase)
        self.assertTrue(
            corrected_phrase.startswith("Thế gian") or corrected_phrase.startswith("Thế giới"),
            f"Expected 'Thế gian' or 'Thế giới', got: {corrected_phrase}"
        )

    def test_uppercase_heading_correction(self):
        heading = "BỘC LẬP - TỰ DO - HẠNH PHÚC"
        expected = "ĐỘC LẬP - TỰ DO - HẠNH PHÚC"
        corrected, count = correct_vietnamese_spelling(heading)
        self.assertEqual(corrected, expected)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
