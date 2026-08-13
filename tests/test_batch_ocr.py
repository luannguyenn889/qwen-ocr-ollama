import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from app.core.batch_ocr import (
    MODEL, clean_markdown, finalize_markdown, link_extracted_images, needs_table_retry,
    needs_vision_retry, normalize_worker_count, page_is_tiled_scan, process_single_pdf,
    repair_markdown_tables,
)


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

    def test_valid_markdown_table_does_not_retry(self):
        markdown = "| Tên | Giá trị |\n|---|---|\n| GDP | 2109 |"
        self.assertFalse(needs_table_retry(markdown))

    def test_valid_markdown_table_is_preserved(self):
        markdown = "| Tên | Giá trị |\n|---|---|\n| GDP | 2109 |"
        self.assertEqual(repair_markdown_tables(markdown), markdown)

    def test_inconsistent_markdown_table_becomes_valid_html(self):
        markdown = (
            "Trước\n\n"
            "| Thị trường | Lượng | Trị giá |\n"
            "|---|---:|---:|\n"
            "| Việt Nam | 10 | 20 | 30 |\n\n"
            "Sau"
        )
        repaired = finalize_markdown(markdown)
        self.assertIn("<table>", repaired)
        self.assertIn("<td>30</td>", repaired)
        self.assertNotIn("| Việt Nam |", repaired)
        self.assertIn("Trước", repaired)
        self.assertIn("Sau", repaired)

    def test_separator_column_mismatch_becomes_html(self):
        markdown = "| A | B |\n|---|\n| 1 | 2 |"
        repaired = repair_markdown_tables(markdown)
        self.assertIn("<th>A</th>", repaired)
        self.assertIn("<td>2</td>", repaired)

    def test_pipe_inside_code_span_does_not_create_extra_cell(self):
        markdown = "| Cú pháp | Mô tả |\n|---|---|\n| `a|b` | lựa chọn |"
        self.assertEqual(repair_markdown_tables(markdown), markdown)

    def test_worker_limits_match_engine_type(self):
        self.assertEqual(normalize_worker_count(4, MODEL), 2)
        self.assertEqual(normalize_worker_count(4, "hybrid"), 2)
        self.assertEqual(normalize_worker_count(4, "PaddleOCR (PP-OCRv6)"), 2)

    def test_missing_and_extra_image_placeholders_are_reported(self):
        linked, remaining = link_extracted_images(
            "Trước ![Hình](image_placeholder.png) sau",
            ["images/one.png", "images/two.png"],
        )
        self.assertIn("images/one.png", linked)
        self.assertEqual(remaining, ["images/two.png"])

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
            output_dir = root / "output"
            (output_dir / "images").mkdir(parents=True)
            (output_dir / "images" / "figure.png").write_bytes(b"image")

            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=["images/figure.png"]),
                patch("app.core.batch_ocr.page_is_tiled_scan", return_value=False),
                patch("app.core.paddle_engine.PaddleOCREngine.ocr_image", return_value="raw text"),
            ):
                result = process_single_pdf(pdf_path, output_dir, client, "hybrid")

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

    def test_quality_failure_retries_only_the_failed_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            image_path = root / "page_1.png"
            image_path.write_bytes(b"image")
            client = Mock()
            client.generate.side_effect = [
                [SimpleNamespace(response="formula_placeholder")],
                [SimpleNamespace(response="Nội dung đã được sửa hợp lệ.")],
            ]

            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=[]),
                patch("app.core.layout_detector.create_layout_detector", return_value=(None, "disabled for test")),
            ):
                result = process_single_pdf(pdf_path, root / "output", client, MODEL)

            self.assertEqual(client.generate.call_count, 2)
            self.assertIn("Nội dung đã được sửa hợp lệ.", result.read_text(encoding="utf-8"))

    def test_column_aware_image_sorting(self):
        from app.core.batch_ocr import extract_images_from_page
        from unittest.mock import MagicMock, patch

        # Mock page image path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_image_path = root / "page_1.png"
            page_image_path.write_bytes(b"mock_png_data")

            # Mock PIL.Image
            mock_img = MagicMock()
            mock_img.width = 1000
            mock_img.height = 1000
            mock_context = mock_img.__enter__.return_value
            mock_context.width = 1000
            mock_context.height = 1000
            
            # Mock layout_detector
            mock_detector = MagicMock()
            # Bounding box 1: column 2, top (y=100) -> center_x = 650
            # Bounding box 2: column 1, bottom (y=500) -> center_x = 100
            mock_detector.detect_layout_tables_and_images.return_value = (
                [],  # no tables
                [(600.0, 100.0, 700.0, 200.0), (50.0, 500.0, 150.0, 600.0)]
            )

            # Columns layout: Column 1 (0 to 450), Column 2 (450 to 1000)
            columns = [(0, 450), (450, 1000)]

            # Mock pymupdf document and page
            mock_page = MagicMock()
            mock_page.rect = MagicMock(width=1000, height=1000)
            mock_doc = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1

            with (
                patch("pymupdf.open", return_value=mock_doc),
                patch("PIL.Image.open", return_value=mock_img)
            ):
                result_paths = extract_images_from_page(
                    pdf_path=Path("dummy.pdf"),
                    page_index=0,
                    output_img_dir=root / "images",
                    prefix="test",
                    layout_detector=mock_detector,
                    page_image_path=page_image_path,
                    segments=[(0, 0, 450, 1000), (450, 0, 1000, 1000)]
                )

                # Since Box 2 (at y=500, col 1) is in Column 1, and Box 1 (at y=100, col 2) is in Column 2,
                # Column 1 items must be sorted first!
                # So layout_img_2 (which is Box 2) should be first, and layout_img_1 should be second.
                self.assertEqual(len(result_paths), 2)
                self.assertIn("layout_img_2.png", result_paths[0])
                self.assertIn("layout_img_1.png", result_paths[1])

    def test_formula_placeholder_replacement(self):
        # Verify that formula_placeholder in markdown is replaced in reading order with recognized LaTeX strings.
        from app.core.batch_ocr import process_single_pdf
        from unittest.mock import MagicMock
        # Mock dependencies
        client = MagicMock()
        
        # We need mock client generate to return stream chunks
        class DummyChunk:
            def __init__(self, response):
                self.response = response
        client.generate.return_value = [
            DummyChunk("This is formula_placeholder and another formula_placeholder.")
        ]
        
        # Mock layout detector to return a dummy formula box
        mock_detector = MagicMock()
        mock_detector.analyse.return_value = ([], False)
        mock_detector.detect_layout_tables_images_and_formulas.return_value = (
            [], [], [(50.0, 100.0, 150.0, 150.0), (200.0, 200.0, 300.0, 250.0)]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("app.core.layout_detector.create_layout_detector", return_value=(mock_detector, "layout enabled for test")),
            patch("app.core.formula_ocr.recognize_formula", side_effect=["E = mc^2", "a^2 + b^2 = c^2"]),
            patch("app.core.batch_ocr.pdf_page_count", return_value=1),
            patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([Path(temp_dir) / "page_1.png"])),
            patch("app.core.batch_ocr.extract_images_from_page", return_value=[])
        ):
            # Write a dummy page_1.png
            from PIL import Image
            img = Image.new("RGB", (1000, 1000), "white")
            img_path = Path(temp_dir) / "page_1.png"
            img.save(img_path)
            
            pdf_path = Path(temp_dir) / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            
            output_dir = Path(temp_dir) / "output"
            result = process_single_pdf(pdf_path, output_dir, client, "qwen3.5:4b")
            
            # Read output markdown
            md_content = result.read_text(encoding="utf-8")
            self.assertIn("$E = mc^2$", md_content)
            self.assertIn("$a^2 + b^2 = c^2$", md_content)


if __name__ == "__main__":
    unittest.main()
