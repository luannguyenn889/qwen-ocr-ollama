import unittest

from app.core.ocr_metrics import score_markdown


class OcrMetricsTests(unittest.TestCase):
    def test_identical_markdown_scores_perfectly(self):
        markdown = """# Tiêu đề

| A | B |
|---|---|
| 1 | 2 |

- mục

$E = mc^2$
"""
        scores = score_markdown(markdown, markdown)
        self.assertEqual(scores["cer"], 0)
        self.assertEqual(scores["wer"], 0)
        self.assertTrue(scores["numbers_exact"])
        self.assertTrue(scores["formulas_exact"])
        self.assertEqual(scores["table_f1"], 1)
        self.assertEqual(scores["list_f1"], 1)
        self.assertTrue(scores["markdown_valid"])

    def test_detects_markdown_syntax_error(self):
        scores = score_markdown("Văn bản", "Văn bản với $latex lỗi")
        self.assertFalse(scores["markdown_valid"])
        self.assertIn("unbalanced_inline_math", scores["markdown_errors"])


if __name__ == "__main__":
    unittest.main()
