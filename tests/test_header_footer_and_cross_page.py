import unittest
from app.core.markdown_normalizer import (
    normalize_structure,
    remove_standalone_page_artifacts,
    remove_repeated_running_headers_and_footers,
    stitch_cross_page_paragraphs,
    MarkdownNormalizationStats,
)


class HeaderFooterAndCrossPageTests(unittest.TestCase):
    def test_remove_standalone_page_number_variants(self):
        text = (
            "<!-- Page 1 -->\n"
            "Nội dung trang đầu.\n"
            "Trang 1/5\n\n"
            "<!-- Page 2 -->\n"
            "- 2 -\n"
            "Nội dung trang hai.\n"
            "Page 2\n\n"
            "<!-- Page 3 -->\n"
            "Nội dung trang ba.\n"
            "[ 3 ]\n\n"
            "<!-- Page 4 -->\n"
            "Nội dung trang bốn.\n"
            "4.\n"
        )
        cleaned = remove_standalone_page_artifacts(text)
        cleaned_lines = [l.strip() for l in cleaned.splitlines()]
        self.assertNotIn("Trang 1/5", cleaned_lines)
        self.assertNotIn("- 2 -", cleaned_lines)
        self.assertNotIn("Page 2", cleaned_lines)
        self.assertNotIn("[ 3 ]", cleaned_lines)
        self.assertNotIn("4.", cleaned_lines)
        self.assertIn("Nội dung trang đầu.", cleaned)
        self.assertIn("Nội dung trang hai.", cleaned)
        self.assertIn("Nội dung trang ba.", cleaned)
        self.assertIn("Nội dung trang bốn.", cleaned)

    def test_remove_repeated_running_headers(self):
        text = (
            "<!-- Page 1 -->\n"
            "UBND HUYỆN YA HỘI\n"
            "Kế hoạch triển khai công tác năm 2026.\n\n"
            "<!-- Page 2 -->\n"
            "UBND HUYỆN YA HỘI\n"
            "Tiếp tục thực hiện các chỉ tiêu đề ra.\n\n"
            "<!-- Page 3 -->\n"
            "UBND HUYỆN YA HỘI\n"
            "Báo cáo tổng kết thi đua."
        )
        stats = MarkdownNormalizationStats()
        cleaned = remove_repeated_running_headers_and_footers(text, stats)
        # Should remove the repeated header from page 2 and page 3
        page_2_start = cleaned.find("<!-- Page 2 -->")
        self.assertNotIn("UBND HUYỆN YA HỘI", cleaned[page_2_start:])
        self.assertGreater(stats.page_artifacts_removed, 0)

    def test_stitch_cross_page_paragraph_continuation(self):
        text = (
            "<!-- Page 1 -->\n"
            "Chính phủ đã ban hành nghị định\n\n"
            "<!-- Page 2 -->\n"
            "về việc điều chỉnh mức lương cơ sở năm 2026."
        )
        stats = MarkdownNormalizationStats()
        stitched = stitch_cross_page_paragraphs(text, stats)
        self.assertIn("Chính phủ đã ban hành nghị định <!-- Page 2 --> về việc điều chỉnh", stitched)
        self.assertEqual(stats.paragraph_lines_joined, 1)

    def test_stitch_cross_page_dehyphenation(self):
        text = (
            "<!-- Page 1 -->\n"
            "Quá trình nghiên-\n\n"
            "<!-- Page 2 -->\n"
            "cứu khoa học và đổi mới sáng tạo."
        )
        stats = MarkdownNormalizationStats()
        stitched = stitch_cross_page_paragraphs(text, stats)
        self.assertIn("Quá trình nghiên <!-- Page 2 --> cứu khoa học", stitched)
        self.assertEqual(stats.paragraph_lines_joined, 1)

    def test_does_not_stitch_across_headings_or_lists(self):
        text = (
            "<!-- Page 1 -->\n"
            "Các nội dung chính bao gồm:\n\n"
            "<!-- Page 2 -->\n"
            "# Mục tiêu thực hiện"
        )
        stats = MarkdownNormalizationStats()
        stitched = stitch_cross_page_paragraphs(text, stats)
        self.assertIn("# Mục tiêu thực hiện", stitched)
        self.assertEqual(stats.paragraph_lines_joined, 0)

    def test_full_normalize_structure_pipeline(self):
        text = (
            "<!-- Page 1 -->\n"
            "SỞ TƯ PHÁP TỈNH\n"
            "Công tác rà soát văn bản quy phạm pháp luật đang được tiến-\n"
            "Trang 1 / 2\n\n"
            "<!-- Page 2 -->\n"
            "SỞ TƯ PHÁP TỈNH\n"
            "hành khẩn trương và nghiêm túc.\n"
            "2/2"
        )
        stats = MarkdownNormalizationStats()
        result = normalize_structure(text, stats)
        self.assertNotIn("Trang 1 / 2", result)
        self.assertNotIn("2/2", result)
        self.assertIn("tiến <!-- Page 2 --> hành", result)


if __name__ == "__main__":
    unittest.main()
