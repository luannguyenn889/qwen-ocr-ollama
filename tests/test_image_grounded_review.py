import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# pyrefly: ignore [missing-import]
from PIL import Image

from app.core.image_grounded_review import (
    SuspiciousLine, find_suspicious_lines, review_suspicious_lines,
    select_review_candidates, should_reread_full_page,
)
from app.core.block_assembler import BlockSpan


class ImageGroundedReviewTests(unittest.TestCase):
    def test_exact_block_span_assigns_line_bbox(self):
        markdown = "Dòng đúng\nngày 4.8.. tháng 6 năm 2007"
        start = markdown.index("ngày")
        bbox = (10, 100, 500, 150)
        candidates = find_suspicious_lines(
            markdown,
            block_spans=[BlockSpan("page_1_block_002", "text", bbox, start, len(markdown))],
        )
        candidate = next(item for item in candidates if item.index == 1)
        self.assertEqual(candidate.bbox, bbox)

    def test_unmapped_line_is_not_automatically_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (400, 400), "white").save(image_path)
            client = Mock()
            original = "ngày 4.8.. tháng 6 năm 2007"
            result = review_suspicious_lines(
                client, "model", image_path, original, root / "crops", block_spans=[],
            )
        self.assertEqual(result, original)
        client.generate.assert_not_called()
    def test_dynamic_review_budget_is_per_severity(self):
        candidates = [
            SuspiciousLine(index, f"line {index}", "unusual_word", "low")
            for index in range(8)
        ] + [
            SuspiciousLine(20 + index, "critical", "mass_diacritic_loss", "critical")
            for index in range(5)
        ]
        selected = select_review_candidates(candidates)
        self.assertEqual(sum(item.severity == "low" for item in selected), 0)
        self.assertEqual(sum(item.severity == "critical" for item in selected), 2)

    def test_multiple_regions_share_one_qwen_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            lines = [
                "Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007",
                "Nội dung kiểm tra vùng thứ hai...",
            ] + [f"Dòng nội dung bình thường số {index}" for index in range(10)]
            client = Mock()
            client.generate.return_value = SimpleNamespace(response=(
                '{"items":['
                '{"id":"line_1","original":"Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007",'
                '"corrected":"Dak Ta Ley, ngày 18 tháng 6 năm 2007","confidence":0.99},'
                '{"id":"line_2","original":"Nội dung kiểm tra vùng thứ hai...",'
                '"corrected":"Nội dung kiểm tra vùng thứ hai","confidence":0.99}'
                ']}'
            ))
            result = review_suspicious_lines(
                client, "model", image_path, "\n".join(lines), root / "crops",
            )
        self.assertIn("ngày 18 tháng 6", result)
        self.assertIn("Nội dung kiểm tra", result)
        self.assertEqual(client.generate.call_count, 1)
        self.assertEqual(len(client.generate.call_args.kwargs["images"]), 1)

    def test_full_page_reread_does_not_continue_with_region_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            original = "\n".join(
                f"Nơi {index}, ngày {index}.8.. tháng 6 năm 2007" for index in range(1, 4)
            )
            with patch("app.core.batch_ocr.ocr_qwen_images", return_value=original) as reread:
                result = review_suspicious_lines(
                    Mock(), "model", image_path, original, root / "crops",
                )
        self.assertEqual(result, original)
        reread.assert_called_once()

    def test_low_spelling_warning_does_not_call_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            warning = [SuspiciousLine(0, "Từ hiếm", "unusual_word", "low")]
            with patch(
                "app.core.image_grounded_review.find_suspicious_lines",
                return_value=warning,
            ):
                result = review_suspicious_lines(
                    client, "model", image_path, "Từ hiếm", root / "crops",
                )
        self.assertEqual(result, "Từ hiếm")
        client.generate.assert_not_called()

    def test_three_high_severity_lines_trigger_full_page_reread(self):
        candidates = [
            SuspiciousLine(index, "broken", "truncated_line", "high")
            for index in range(3)
        ]
        self.assertTrue(should_reread_full_page(candidates, "a\nb\nc\nd\ne\nf"))

    def test_one_suspicious_line_stays_a_regional_reread(self):
        candidates = [SuspiciousLine(0, "broken", "malformed_date", "high")]
        self.assertFalse(should_reread_full_page(candidates, "broken"))

    def test_graphic_overlap_and_regional_disagreement_are_candidates(self):
        markdown = "Dòng thứ nhất\nDòng thứ hai"
        candidates = find_suspicious_lines(
            markdown,
            graphic_regions=[(20, 10, 80, 35)],
            regional_text_by_index={1: "Nội dung hoàn toàn khác"},
            block_spans=[
                BlockSpan("block_1", "text", (0, 0, 100, 40), 0, len("Dòng thứ nhất")),
                BlockSpan(
                    "block_2", "text", (0, 50, 100, 90), markdown.index("Dòng thứ hai"),
                    len(markdown),
                ),
            ],
        )
        reasons = {item.index: item.reason for item in candidates}
        self.assertIn("graphic_overlap", reasons[0])
        self.assertIn("ocr_disagreement", reasons[1])
        self.assertEqual(next(item for item in candidates if item.index == 0).bbox, (0, 0, 100, 40))

    def test_detects_malformed_date_but_not_valid_date(self):
        markdown = (
            "Hà Nội, ngày 4.8.. tháng 6 năm 2007\n"
            "Hà Nội, ngày 25 tháng 5 năm 2007"
        )
        candidates = find_suspicious_lines(markdown)
        self.assertEqual([item.index for item in candidates], [0])
        self.assertIn("malformed_date", candidates[0].reason)

    def test_footer_review_does_not_require_a_known_signing_title(self):
        candidates = find_suspicious_lines(
            "Biên bản chuyên môn\nVAI TRÒ TỰ DO\nDòng cuối",
            review_document_footer=True,
        )
        self.assertTrue(any("possible_missing_printed_footer" in item.reason for item in candidates))

    def test_detects_missing_signing_block_after_recipient_section(self):
        candidates = find_suspicious_lines(
            "HỘI ĐỒNG NHÂN DÂN\nNội dung\n\n**Nơi nhận:**\n- Lưu VP.",
            review_document_footer=True,
        )
        missing_block = next(
            item for item in candidates if "possible_missing_printed_footer" in item.reason
        )
        self.assertEqual(missing_block.text, "- Lưu VP.")

    def test_administrative_footer_review_does_not_depend_on_known_titles(self):
        candidates = find_suspicious_lines(
            "HỘI ĐỒNG NHÂN DÂN\nSố: 01/BC-HĐND\nKhông có mục nơi nhận\nNGƯỜI XÁC NHẬN HIỆN TRƯỜNG",
            review_document_footer=True,
        )
        self.assertTrue(any("possible_missing_printed_footer" in item.reason for item in candidates))

    def test_appends_image_confirmed_printed_signing_block_after_recipients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            client.generate.return_value = SimpleNamespace(response=(
                '{"original":"- Lưu VP.",'
                '"printed_lines":["TM. HỘI ĐỒNG NHÂN DÂN XÃ",'
                '"CHỦ TỊCH","PHAN VĂN CƯỜNG"],'
                '"confidence":0.99}'
            ))
            result = review_suspicious_lines(
                client, "model", image_path,
                "HỘI ĐỒNG NHÂN DÂN\nNội dung\n\n**Nơi nhận:**\n- Lưu VP.",
                root / "crops", review_document_footer=True,
                regions=[(0, 900, 700, 1100)],
                graphic_regions=[(300, 950, 650, 1150)],
            )
        self.assertTrue(result.endswith("CHỦ TỊCH\nPHAN VĂN CƯỜNG"))

    def test_applies_only_high_confidence_image_confirmed_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            client.generate.return_value = SimpleNamespace(response=(
                '{"original":"Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007",'
                '"corrected":"Dak Ta Ley, ngày 18 tháng 6 năm 2007","confidence":0.99}'
            ))
            result = review_suspicious_lines(
                client, "model", image_path,
                "Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007", root / "crops",
            )
        self.assertIn("ngày 18 tháng 6", result)

    def test_keeps_low_confidence_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            client.generate.return_value = SimpleNamespace(response=(
                '{"original":"Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007",'
                '"corrected":"Dak Ta Ley, ngày 18 tháng 6 năm 2007","confidence":0.9}'
            ))
            original = "Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007"
            result = review_suspicious_lines(client, "model", image_path, original, root / "crops")
        self.assertEqual(result, original)

    def test_quality_100_skips_line_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            original = "Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007"
            result = review_suspicious_lines(
                client, "model", image_path, original, root / "crops", quality_score=100.0,
            )
        self.assertEqual(result, original)
        client.generate.assert_not_called()

    def test_footer_without_graphic_text_overlap_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            original = "Nội dung\nNGƯỜI XÁC NHẬN"
            result = review_suspicious_lines(
                client, "model", image_path, original, root / "crops",
                review_document_footer=True,
                regions=[(0, 900, 300, 1050)],
                graphic_regions=[(500, 900, 750, 1100)],
            )
        self.assertEqual(result, original)
        client.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
