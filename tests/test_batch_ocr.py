import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from app.core.batch_ocr import MODEL, clean_markdown, needs_vision_retry, page_is_tiled_scan, process_single_pdf


class BatchOcrTests(unittest.TestCase):
    def test_prose_is_not_left_inside_math_delimiters(self):
        malformed = "$2109 đô la Mỹ, và giá trị HDI là 0,666 đứng thứ 116 trong số 188 nước năm 2015.$"
        cleaned = clean_markdown(malformed)
        self.assertNotIn("$", cleaned)
        self.assertIn("2109 đô la Mỹ", cleaned)

    def test_unclosed_math_marker_is_removed_from_prose(self):
        malformed = "$2109 đô la Mỹ, và giá trị HDI là 0,666 đứng thứ 116 trong số 188 nước"
        self.assertNotIn("$", clean_markdown(malformed))

    def test_low_diacritic_vietnamese_requests_vision_retry(self):
        malformed = "Viet Nam la nuoc co muc thu nhap trung binh va chuong trinh trong nam. " * 12
        self.assertTrue(needs_vision_retry(malformed))

    def test_tiled_scan_is_not_treated_as_many_figures(self):
        pdf_path = Path(__file__).resolve().parent.parent / "PDF" / "de-van.pdf"
        if not pdf_path.exists():
            self.skipTest("local tiled-scan fixture is unavailable")
        self.assertTrue(page_is_tiled_scan(pdf_path, 0))

    def test_failed_page_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            output_dir = root / "output"
            output_dir.mkdir()
            output_path = output_dir / "input.md"
            output_path.write_text("previous result", encoding="utf-8")
            image_path = root / "page_1.png"
            image_path.write_bytes(b"image")

            client = Mock()
            client.generate.side_effect = RuntimeError("Ollama unavailable")

            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=[]),
                patch("app.core.paddle_engine.PaddleOCREngine.ocr_image", side_effect=RuntimeError("Paddle unavailable")),
            ):
                with self.assertRaisesRegex(RuntimeError, "existing output was preserved"):
                    process_single_pdf(pdf_path, output_dir, client, MODEL)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous result")
            self.assertFalse((output_dir / "input.md.tmp").exists())

    def test_hybrid_alias_uses_default_qwen_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            image_path = root / "page_1.png"
            image_path.write_bytes(b"image")

            client = Mock()
            client.generate.return_value = [SimpleNamespace(response="# Result")]

            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=["images/figure.png"]),
                patch("app.core.batch_ocr.page_is_tiled_scan", return_value=False),
                patch("app.core.paddle_engine.PaddleOCREngine.ocr_image", return_value="raw text"),
            ):
                result = process_single_pdf(pdf_path, root / "output", client, "hybrid")

            self.assertEqual(client.generate.call_args.kwargs["model"], MODEL)
            self.assertEqual(client.generate.call_args.kwargs["images"], [str(image_path)])
            self.assertEqual(
                result.read_text(encoding="utf-8"),
                "<!-- Page 1 -->\n\n# Result\n\n![Hình ảnh trang 1](images/figure.png)",
            )

    def test_two_workers_keep_original_page_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            images = []
            for page_number in (1, 2):
                image_path = root / f"page_{page_number}.png"
                image_path.write_bytes(b"image")
                images.append(image_path)

            def generate(**kwargs):
                page_number = Path(kwargs["images"][0]).stem.split("_")[1]
                if page_number == "1":
                    time.sleep(0.01)
                return [SimpleNamespace(response=f"Page {page_number}")]

            client = Mock()
            client.generate.side_effect = generate
            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=2),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter(images)),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=[]),
            ):
                result = process_single_pdf(pdf_path, root / "output", client, MODEL, workers=2)

            markdown = result.read_text(encoding="utf-8")
            self.assertLess(markdown.index("Page 1"), markdown.index("Page 2"))


if __name__ == "__main__":
    unittest.main()
