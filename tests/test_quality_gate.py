import tempfile
import unittest
from pathlib import Path

from app.core.quality_gate import QualityReport, choose_best_page, evaluate_page, validate_page_numbers


class QualityGateTests(unittest.TestCase):
    def test_detects_requested_page_quality_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            markdown = (
                "$công thức chưa đóng\n"
                "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n"
                "image_placeholder.png\n"
                "![Thiếu](images/missing.png)\n"
                + "motchuoitubidinhkhongcokhoangtrang" * 3
            )
            errors = evaluate_page(markdown, directory).errors
        self.assertIn("glued_words", errors)
        self.assertIn("unbalanced_math_delimiters", errors)
        self.assertTrue(any(error.startswith("malformed_markdown_table") for error in errors))
        self.assertIn("unresolved_placeholder", errors)
        self.assertIn("missing_image:images/missing.png", errors)

    def test_existing_relative_image_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "ok.png").write_bytes(b"image")
            report = evaluate_page("Nội dung hợp lệ.\n\n![Ảnh](images/ok.png)", root)
        self.assertTrue(report.passed)

    def test_long_image_filename_is_not_a_glued_word(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            filename = "thong_tin_du_lich_thang_2_2024_page_1_layout_img_2.png"
            (root / "images" / filename).write_bytes(b"image")
            report = evaluate_page(f"![Ảnh](images/{filename})", root)
        self.assertNotIn("glued_words", report.errors)

    def test_placeholder_is_not_also_reported_as_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_page("![Ảnh](image_placeholder.png)", directory)
        self.assertIn("unresolved_placeholder", report.errors)
        self.assertFalse(any(error.startswith("missing_image:") for error in report.errors))

    def test_high_confidence_unaccented_vietnamese_is_an_error(self):
        paragraph = (
            "Viet Nam la mot nuoc co nhieu diem du lich va cac chuong trinh phat trien "
            "duoc thuc hien trong nam nay cho nguoi dan va doanh nghiep tai thi truong. "
        ) * 2
        report = evaluate_page(paragraph, ".")
        self.assertIn("missing_vietnamese_diacritics", report.errors)

    def test_english_paragraph_is_not_reported_as_missing_vietnamese(self):
        paragraph = (
            "This is an English report about the tourism market and the activities that "
            "were completed by the organization for visitors from several countries. "
        ) * 2
        report = evaluate_page(paragraph, ".")
        self.assertNotIn("missing_vietnamese_diacritics", report.errors)
        self.assertNotIn("suspected_missing_vietnamese_diacritics", report.warnings)

    def test_numeric_table_is_excluded_from_diacritic_check(self):
        table = "| Nam | 2022 | 2023 |\n|---|---:|---:|\n" + "\n".join(
            f"| Thi truong {number} | {number * 10} | {number * 20} |" for number in range(20)
        )
        report = evaluate_page(table, ".")
        self.assertNotIn("missing_vietnamese_diacritics", report.errors)

    def test_missing_page_is_detected(self):
        report = validate_page_numbers([1, 3], 3)
        self.assertIn("missing_pages:2", report.errors)

    def test_best_page_prefers_fewer_quality_errors(self):
        chosen, report = choose_best_page(
            "longer but broken text",
            QualityReport(("glued_words", "unresolved_placeholder")),
            "short retry",
            QualityReport(("glued_words",)),
        )
        self.assertEqual("short retry", chosen)
        self.assertEqual(("glued_words",), report.errors)

    def test_best_page_uses_content_length_as_tiebreaker(self):
        chosen, _ = choose_best_page(
            "short",
            QualityReport(("glued_words",)),
            "a more complete retry",
            QualityReport(("glued_words",)),
        )
        self.assertEqual("a more complete retry", chosen)

    def test_best_page_uses_error_severity_not_raw_count(self):
        chosen, _ = choose_best_page(
            "retry with two minor issues",
            QualityReport(("glued_words", "unbalanced_math_delimiters")),
            "original with severe issue",
            QualityReport(("empty_page",)),
        )
        self.assertEqual("retry with two minor issues", chosen)

    def test_long_identifier_is_not_a_glued_word(self):
        report = evaluate_page("CustomerAccountTransactionReferenceIdentifier", ".")
        self.assertNotIn("glued_words", report.errors)


if __name__ == "__main__":
    unittest.main()
