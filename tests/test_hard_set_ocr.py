"""Unit tests for Hard Set OCR features: checkboxes, circled choices, strikethrough, and symbols."""

import unittest
from app.core.markdown_normalizer import (
    normalize_checkboxes,
    normalize_special_symbols,
    normalize_structure,
)
from app.core.math_cleanup import normalize_answer_math


class TestHardSetOCR(unittest.TestCase):
    def test_checkbox_normalization_list(self):
        raw = """
- [x] Đã hoàn thành
- [ ] Chưa hoàn thành
- [v] Đã xác nhận
- [V] Đã nộp hồ sơ
- [*] Đã thanh toán
☑ Nam
☐ Nữ
- [ ] Khác
"""
        normalized = normalize_checkboxes(raw)
        self.assertIn("- [x] Đã hoàn thành", normalized)
        self.assertIn("- [ ] Chưa hoàn thành", normalized)
        self.assertIn("- [x] Đã xác nhận", normalized)
        self.assertIn("- [x] Đã nộp hồ sơ", normalized)
        self.assertIn("- [x] Đã thanh toán", normalized)
        self.assertIn("- [x] Nam", normalized)
        self.assertIn("- [ ] Nữ", normalized)
        self.assertIn("- [ ] Khác", normalized)

    def test_inline_checkbox_normalization(self):
        raw = "Giới tính: [v] Nam  [ ] Nữ  [X] Không xác định"
        normalized = normalize_checkboxes(raw)
        self.assertEqual(normalized, "Giới tính: [x] Nam  [ ] Nữ  [x] Không xác định")

    def test_special_symbols_temperatures_and_dimensions(self):
        raw = "Nhiệt độ bảo quản: 37 oC (hoặc 37.5 0C, 100 °C), kích thước 200 x 300 mm, nồng độ 50 ug/ml."
        normalized = normalize_special_symbols(raw)
        self.assertIn("37°C", normalized)
        self.assertIn("37.5°C", normalized)
        self.assertIn("100°C", normalized)
        self.assertIn("200 × 300 mm", normalized)
        self.assertIn("50 µg/ml", normalized)

    def test_legal_and_comparison_symbols(self):
        raw = "Bản quyền (c) 2026, Thương hiệu (tm), Đăng ký (R). Độ sai số +-5%, n <= 100, x >= 50, y ~= 10."
        normalized = normalize_special_symbols(raw)
        self.assertIn("© 2026", normalized)
        self.assertIn("™", normalized)
        self.assertIn("®", normalized)
        self.assertIn("±5%", normalized)
        self.assertIn("n ≤ 100", normalized)
        self.assertIn("x ≥ 50", normalized)
        self.assertIn("y ≈ 10", normalized)

    def test_circled_choices_and_strikethrough_preservation(self):
        raw = """
Câu 1: Thủ đô của Việt Nam là gì?
**(A)** Hà Nội
B. TP. Hồ Chí Minh
C. Đà Nẵng
D. Hải Phòng

Ghi chú hợp đồng:
Điều 1: ~~Thời hạn hợp đồng 1 năm~~ Thời hạn hợp đồng 2 năm.
"""
        normalized = normalize_structure(raw)
        self.assertIn("**(A)** Hà Nội", normalized)
        self.assertIn("~~Thời hạn hợp đồng 1 năm~~", normalized)

    def test_image_paths_with_spaces_normalization(self):
        raw = "![Figure 1](images/Vật lý 1_HK1 25-26 final_page_1_layout_img_1.png)"
        normalized = normalize_structure(raw)
        self.assertEqual(
            normalized,
            "![Figure 1](<images/Vật lý 1_HK1 25-26 final_page_1_layout_img_1.png>)",
        )

    def test_inline_phrase_repetition_collapse(self):
        raw = "Các lễ hội như Tết Nguyên Đán, Lễ hội Đền Mẫu Thượng Ngàn, Lễ hội Đền Mẫu Thượng Ngàn, Lễ hội Đền Mẫu Thượng Ngàn, Lễ hội Đền Mẫu Thượng"
        normalized = normalize_structure(raw)
        self.assertNotIn("Lễ hội Đền Mẫu Thượng Ngàn, Lễ hội Đền Mẫu Thượng Ngàn", normalized)
        self.assertIn("Lễ hội Đền Mẫu Thượng Ngàn", normalized)

    def test_remove_scanner_zoom_artifacts(self):
        raw = """<!-- Page 1 -->

Nội dung trang 1.

<!-- Page 2 -->

# 100%
"""
        normalized = normalize_structure(raw)
        self.assertNotIn("100%", normalized)
        self.assertIn("Nội dung trang 1.", normalized)

    def test_orphan_think_and_pipes_normalization(self):
        raw = """Số: 57/TTr - UBND | Ya Hội, ngày 28 tháng 6 năm 2016

Nội dung văn bản.
</think>
"""
        normalized = normalize_structure(raw)
        self.assertNotIn("</think>", normalized)
        self.assertIn("Số: 57/TTr - UBND\nYa Hội, ngày 28 tháng 6 năm 2016", normalized)

    def test_remove_empty_pipe_rows(self):
        raw = """
Đơn vị
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

Dự toán năm 2025 (Đơn vị tính: đồng)
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
"""
        normalized = normalize_structure(raw)
        self.assertNotIn("| | |", normalized)
        self.assertIn("Đơn vị", normalized)
        self.assertIn("Dự toán năm 2025", normalized)


if __name__ == "__main__":
    unittest.main()
