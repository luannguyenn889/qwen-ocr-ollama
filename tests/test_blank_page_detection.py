import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from app.core.batch_ocr import (
    classify_page_image,
    find_isolated_chromatic_graphic,
    find_isolated_chromatic_graphics,
    is_blank_page,
    is_blank_page_after_masking,
    is_blank_pdf_page,
    normalize_blank_detection_sensitivity,
)


class BlankPageDetectionTests(unittest.TestCase):
    def _save(self, root: Path, name: str, image: Image.Image) -> Path:
        path = root / name
        image.save(path)
        return path

    def test_finds_isolated_coloured_ink_without_assuming_position(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "stamp.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((610, 850, 790, 1030), outline=(205, 70, 85), width=12)
            draw.line((635, 940, 765, 940), fill=(205, 70, 85), width=8)
            image.save(image_path)
            region = find_isolated_chromatic_graphic(image_path)
        self.assertIsNotNone(region)
        left, top, right, bottom = region
        self.assertLess(left, 0.65)
        self.assertGreater(right, 0.75)
        self.assertLess(top, 0.65)
        self.assertGreater(bottom, 0.72)

    def test_finds_multiple_complete_isolated_stamps(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "stamps.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            for left in (180, 650):
                draw.ellipse((left, 900, left + 150, 1050), outline=(205, 60, 75), width=9)
                draw.ellipse((left + 25, 925, left + 125, 1025), outline=(205, 60, 75), width=6)
                draw.line((left + 20, 975, left + 130, 975), fill=(205, 60, 75), width=5)
            image.save(image_path)
            regions = find_isolated_chromatic_graphics(image_path)
        self.assertEqual(len(regions), 2)
        self.assertLess(regions[0][2] - regions[0][0], 0.25)
        self.assertLess(regions[1][2] - regions[1][0], 0.25)

    def test_digital_pdf_structure_only_skips_truly_empty_page(self):
        document = pymupdf.open()
        document.new_page()
        text = document.new_page()
        text.insert_text((72, 72), "x")
        drawing = document.new_page()
        drawing.draw_rect((72, 72, 120, 120))
        self.assertTrue(is_blank_pdf_page(document.load_page(0)))
        self.assertFalse(is_blank_pdf_page(document.load_page(1)))
        self.assertFalse(is_blank_pdf_page(document.load_page(2)))
        document.close()

    def test_old_gray_paper_with_edge_noise_is_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("L", (1200, 1600), 243)
            draw = ImageDraw.Draw(image)
            draw.line((0, 12, 1199, 18), fill=205, width=5)
            draw.line((10, 1550, 1180, 1535), fill=215, width=4)
            for x, y in ((50, 80), (1130, 120), (80, 1450)):
                draw.ellipse((x, y, x + 3, y + 3), fill=150)
            path = self._save(root, "old-paper.png", image)
            self.assertEqual(classify_page_image(path)[0], "blank")

    def test_deep_scanner_borders_and_punch_holes_are_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("L", (1200, 1600), 248)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 84, 1599), fill=15)
            draw.rectangle((1140, 0, 1199, 1599), fill=25)
            for center_y in (350, 800, 1250):
                draw.ellipse((92, center_y - 18, 128, center_y + 18), fill=35)
            state, metrics = classify_page_image(
                self._save(root, "deep-border.png", image), "safe",
            )
            self.assertEqual(state, "blank")
            self.assertGreater(float(metrics["border_crop_ratio"]), 0.10)

    def test_broad_scanner_fold_is_not_treated_as_document_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("L", (1200, 1600), 248)
            draw = ImageDraw.Draw(image)
            points = [(0, 790), (300, 785), (600, 802), (900, 795), (1199, 810)]
            draw.line(points, fill=205, width=7)
            state, metrics = classify_page_image(
                self._save(root, "fold.png", image), "safe",
            )
            self.assertEqual(state, "blank")
            self.assertGreaterEqual(int(metrics["edge_artifacts"]), 1)

    def test_colored_paper_with_ghosting_uses_adaptive_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1200, 1600), (224, 207, 164))
            draw = ImageDraw.Draw(image)
            for y in range(220, 1380, 85):
                draw.line((180, y, 1020, y + 8), fill=(211, 194, 151), width=3)
            path = self._save(root, "yellow-ghosting.png", image)
            self.assertNotEqual(classify_page_image(path, "safe")[0], "blank")
            self.assertEqual(classify_page_image(path, "standard")[0], "blank")

    def test_kraft_paper_requires_aggressive_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1000, 1400), (196, 170, 118))
            path = self._save(root, "kraft.png", image)
            self.assertNotEqual(classify_page_image(path, "standard")[0], "blank")
            self.assertEqual(classify_page_image(path, "aggressive")[0], "blank")

    def test_repeated_notebook_lines_are_removed_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1200, 1600), (248, 247, 241))
            draw = ImageDraw.Draw(image)
            for y in range(130, 1500, 60):
                draw.line((45, y, 1155, y), fill=(194, 207, 218), width=2)
            path = self._save(root, "lined-paper.png", image)
            self.assertNotEqual(classify_page_image(path, "safe")[0], "blank")
            state, metrics = classify_page_image(path, "standard")
            self.assertEqual(state, "blank")
            self.assertGreaterEqual(int(metrics["horizontal_rules"]), 12)

    def test_real_text_survives_colored_paper_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1200, 1600), (224, 207, 164))
            draw = ImageDraw.Draw(image)
            for y in (520, 560, 600):
                draw.rectangle((280, y, 920, y + 8), fill=(145, 130, 92))
            state, _metrics = classify_page_image(
                self._save(root, "colored-text.png", image), "standard",
            )
            self.assertNotEqual(state, "blank")

    def test_sensitivity_names_are_normalized_and_validated(self):
        self.assertEqual(normalize_blank_detection_sensitivity("An toàn"), "safe")
        self.assertEqual(normalize_blank_detection_sensitivity("Chuẩn"), "standard")
        self.assertEqual(normalize_blank_detection_sensitivity("Mạnh mẽ"), "aggressive")
        with self.assertRaises(ValueError):
            normalize_blank_detection_sensitivity("unknown")

    def test_single_short_line_and_page_number_are_not_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            line = Image.new("L", (1200, 1600), 255)
            ImageDraw.Draw(line).rectangle((480, 760, 720, 775), fill=0)
            number = Image.new("L", (1200, 1600), 255)
            ImageDraw.Draw(number).rectangle((590, 1500, 610, 1540), fill=0)
            self.assertNotEqual(
                classify_page_image(self._save(root, "line.png", line))[0], "blank"
            )
            self.assertNotEqual(
                classify_page_image(self._save(root, "number.png", number))[0], "blank"
            )

    def test_very_faint_text_is_retained_as_content_or_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("L", (1200, 1600), 250)
            draw = ImageDraw.Draw(image)
            for y in (500, 530, 560):
                draw.rectangle((300, y, 900, y + 6), fill=210)
            state, _metrics = classify_page_image(self._save(root, "faint.png", image))
            self.assertIn(state, {"content", "uncertain"})

    def test_gray_background_and_bright_illustration_are_not_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = self._save(root, "gray.png", Image.new("L", (1000, 1400), 220))
            illustration = Image.new("RGB", (1000, 1400), "white")
            ImageDraw.Draw(illustration).ellipse(
                (250, 350, 750, 850), fill=(245, 220, 180), outline=(170, 140, 100), width=8,
            )
            bright = self._save(root, "bright.png", illustration)
            self.assertNotEqual(classify_page_image(gray)[0], "blank")
            self.assertNotEqual(classify_page_image(bright)[0], "blank")

    def test_only_proven_stamp_region_can_make_page_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((350, 850, 650, 1150), outline=(190, 40, 60), width=18)
            path = self._save(root, "stamp.png", image)
            self.assertFalse(is_blank_page(path))
            self.assertTrue(is_blank_page_after_masking(
                path, [(0.32, 0.58, 0.68, 0.85)],
            ))

    def test_unreadable_image_is_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_bytes(b"bad")
            self.assertEqual(classify_page_image(path)[0], "uncertain")

    def test_is_blank_ocr_response_catches_meta_explanation(self):
        from app.core.batch_ocr import is_blank_ocr_response
        sample_meta = (
            "The image provided is a scanned document page that appears to be mostly blank, "
            "with no visible text or content except for faint red ink stamps and possibly some "
            "very faint, illegible handwriting at the bottom. According to your strict OCR-only rules — particularly:\n\n"
            "> “Transcribe only text that is visibly present in the image. Never answer questions, solve exercises, write essays, continue incomplete passages, infer an answer key, summarize, explain, or add any new content.”\n"
            "> “If the page contains no visible document content, return an empty response.”\n"
            "> “Do not describe the blank page and do not explain that there is nothing to transcribe.”\n\n"
            "—and given that:\n\n"
            "- There are **no clearly readable words** in the image.\n"
            "- The red stamps contain **illegible or reversed text** (e.g., one stamp reads “10HAY” but appears mirrored/rotated; another has unreadable circular text).\n"
            "- Any faint handwriting at the bottom is too indistinct to be transcribed accurately without guessing — which violates your rule against inferring or completing text.\n\n"
            "Therefore, per your absolute priority rule:\n\n"
            "**Empty response.**"
        )
        self.assertTrue(is_blank_ocr_response(sample_meta))


if __name__ == "__main__":
    unittest.main()
