import unittest

from app.core.vietnamese_spell_corrector import correct_vietnamese_spelling


class VietnameseSpellCorrectorTests(unittest.TestCase):
    def test_corrects_unique_bigram_candidate(self):
        corrected, count = correct_vietnamese_spelling("Kết quả nghiên cưu đã hoàn thành.")
        self.assertEqual(corrected, "Kết quả nghiên cứu đã hoàn thành.")
        self.assertEqual(count, 1)

    def test_preserves_all_uppercase_special_text(self):
        corrected, count = correct_vietnamese_spelling("NGHIEN CUU và Nghien cuu")
        self.assertEqual(corrected, "NGHIEN CUU và Nghiên cứu")
        self.assertEqual(count, 2)

    def test_preserves_titlecase_name_pair(self):
        text = "Nghien Cuu ký xác nhận."
        self.assertEqual(correct_vietnamese_spelling(text), (text, 0))

    def test_keeps_already_valid_words(self):
        corrected, count = correct_vietnamese_spelling("nghiên cứu khoa học")
        self.assertEqual(corrected, "nghiên cứu khoa học")
        self.assertEqual(count, 0)

    def test_masks_sensitive_regions(self):
        text = (
            "nghien cuu `nghien cuu` $nghien cuu$ "
            "[nghien cuu](https://example.com/nghien-cuu) "
            "<table><tr><td>nghien cuu</td></tr></table> "
            "121/2026/QĐ-UBND user@example.com DT-2026-01"
        )
        corrected, count = correct_vietnamese_spelling(text)
        self.assertTrue(corrected.startswith("nghiên cứu `nghien cuu` $nghien cuu$"))
        self.assertIn("<td>nghien cuu</td>", corrected)
        self.assertIn("121/2026/QĐ-UBND", corrected)
        self.assertIn("user@example.com", corrected)
        self.assertIn("DT-2026-01", corrected)
        self.assertEqual(count, 2)

    def test_masks_nested_html_table(self):
        text = "<table><tr><td><table><tr><td>nghien cuu</td></tr></table></td></tr></table>"
        self.assertEqual(correct_vietnamese_spelling(text), (text, 0))

    def test_masks_pipe_table(self):
        text = "| Nội dung |\n|---|\n| nghien cuu |"
        self.assertEqual(correct_vietnamese_spelling(text), (text, 0))

    def test_does_not_join_context_across_paragraphs(self):
        text = "nghien\n\ncuu"
        self.assertEqual(correct_vietnamese_spelling(text), (text, 0))


if __name__ == "__main__":
    unittest.main()
