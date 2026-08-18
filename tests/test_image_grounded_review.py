import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from app.core.image_grounded_review import (
    find_suspicious_lines, review_suspicious_lines,
)


class ImageGroundedReviewTests(unittest.TestCase):
    def test_detects_malformed_date_but_not_valid_date(self):
        markdown = (
            "Dak Ta Ley, ngày 4.8.. tháng 6 năm 2007\n"
            "Dak Ta Ley, ngày 25 tháng 5 năm 2007"
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


if __name__ == "__main__":
    unittest.main()
