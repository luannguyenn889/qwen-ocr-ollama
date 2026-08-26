import tempfile
import time
import unittest
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from app.core.batch_ocr import (
    MODEL, PROMPT, TEXT_ONLY_OUTPUT, QUALITY_RETRY_INSTRUCTION, TABLE_STRUCTURE_REPAIR_PROMPT,
    clean_markdown, detect_and_rotate_page, finalize_markdown, generate_with_retry, link_extracted_images, merge_markdown_tables, needs_table_retry,
    is_blank_page, is_blank_page_after_masking, needs_vision_retry, normalize_escaped_image_links, normalize_worker_count, output_markdown_path, page_is_tiled_scan, process_single_pdf,
    quality_retry_instruction, repair_markdown_tables,
)


class BatchOcrTests(unittest.TestCase):
    def test_blank_page_detection_handles_white_and_off_white_scans(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            white = root / "white.png"
            off_white = root / "off_white.png"
            Image.new("RGB", (1200, 1600), "white").save(white)
            noisy = Image.new("RGB", (1200, 1600), (242, 242, 240))
            draw = ImageDraw.Draw(noisy)
            draw.line((0, 30, 1199, 34), fill=(215, 215, 210), width=3)
            draw.line((20, 1500, 1180, 1490), fill=(220, 218, 215), width=3)
            for x, y in ((80, 90), (1100, 160), (1040, 1490), (60, 1380)):
                draw.ellipse((x, y, x + 4, y + 4), fill=(120, 110, 100))
            noisy.save(off_white)
            self.assertTrue(is_blank_page(white))
            self.assertTrue(is_blank_page(off_white))

    def test_blank_page_detection_protects_sparse_real_content(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sparse.png"
            image = Image.new("L", (1200, 1600), 255)
            ImageDraw.Draw(image).rectangle((400, 780, 800, 795), fill=0)
            image.save(image_path)
            self.assertFalse(is_blank_page(image_path))

    def test_invalid_image_is_never_silently_classified_as_blank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "invalid.png"
            image_path.write_bytes(b"not-an-image")
            self.assertFalse(is_blank_page(image_path))

    def test_page_with_only_masked_signature_or_stamp_becomes_blank(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "stamp_only.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((350, 850, 650, 1150), outline=(180, 30, 50), width=18)
            draw.line((400, 1020, 600, 900), fill=(30, 30, 30), width=12)
            image.save(image_path)
            region = [(0.32, 0.58, 0.68, 0.85)]
            self.assertFalse(is_blank_page(image_path))
            self.assertTrue(is_blank_page_after_masking(image_path, region))

    def test_masking_stamp_does_not_hide_independent_body_content(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "stamp_and_text.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((350, 850, 650, 1150), outline=(180, 30, 50), width=18)
            draw.rectangle((150, 250, 850, 285), fill="black")
            image.save(image_path)
            self.assertFalse(is_blank_page_after_masking(
                image_path, [(0.32, 0.58, 0.68, 0.85)],
            ))

    def test_blank_page_skips_qwen_and_markdown_page_block(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            image_path = root / "page_1.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            client = Mock()
            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.ENABLE_LAYOUT_DETECTION", False),
            ):
                result = process_single_pdf(pdf_path, root / "output", client, MODEL)
            client.generate.assert_not_called()
            self.assertEqual(result.read_text(encoding="utf-8"), "")

    def test_qwen_verbose_blank_page_produces_no_text_marker_or_asset(self):
        from PIL import Image

        response = (
            "The image provided is a scanned document page that appears to be mostly "
            "blank, with very faint and possibly illegible text at the bottom. There "
            "are two red circular stamps visible near the bottom center of the page. "
            "One stamp contains the word 10HAY in reverse, indicating it was stamped "
            "through paper, and the other has indistinct markings. Below these stamps, "
            "there is extremely faint handwritten-style text that is too blurry to "
            "transcribe accurately. Per your instructions: If the page contains no "
            "visible document content, return an empty response. Do not describe the "
            "blank page and do not explain that there is nothing to transcribe. Given "
            "that the only potentially readable elements are minimal, reversed, or "
            "ambiguous and do not constitute clear transcribable document content under "
            "the stated OCR rules, I will return an empty response as required."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "input.pdf"
            pdf_path.write_bytes(b"pdf")
            image_path = root / "page_1.png"
            Image.new("RGB", (800, 1200), "white").save(image_path)
            output_dir = root / "output"
            asset = output_dir / "images" / "page_1_stamp.png"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"temporary extracted graphic")
            client = Mock()
            client.generate.return_value = [SimpleNamespace(response=response)]
            uncertain_metrics = {
                "light_ratio": 0.99, "background_level": 250.0,
                "adaptive_ink_ratio": 0.001, "stddev": 2.0,
                "components": 2, "horizontal_rules": 0, "vertical_rules": 0,
            }

            with (
                patch("app.core.batch_ocr.pdf_page_count", return_value=1),
                patch("app.core.batch_ocr.iter_render_pdf_to_images", return_value=iter([image_path])),
                patch("app.core.batch_ocr.classify_page_image", return_value=("uncertain", uncertain_metrics)),
                patch("app.core.batch_ocr.orient_page_file", return_value=0),
                patch("app.core.batch_ocr.extract_images_from_page", return_value=["images/page_1_stamp.png"]),
                patch("app.core.layout_detector.create_layout_detector", return_value=(None, "disabled for test")),
            ):
                result = process_single_pdf(pdf_path, output_dir, client, MODEL)

            self.assertEqual(client.generate.call_count, 1)
            self.assertEqual(result.read_bytes(), b"")
            self.assertFalse(asset.exists())

    def test_repairs_escaped_image_parenthesis_and_underscores_only(self):
        markdown = (
            r"    ![Ảnh minh họa]\(images/tai\_lieu\_page\_1.png)" "\n"
            r"Công thức giữ nguyên: $x\_1$"
        )
        repaired = normalize_escaped_image_links(markdown)
        self.assertIn("![Ảnh minh họa](images/tai_lieu_page_1.png)", repaired)
        self.assertNotIn("    ![", repaired)
        self.assertIn(r"$x\_1$", repaired)

    def test_detect_and_rotate_page_applies_classifier_angle(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "landscape.png"
            Image.new("RGB", (80, 40), "white").save(image_path)
            with patch("app.core.batch_ocr.detect_text_orientation", return_value=90):
                oriented = detect_and_rotate_page(image_path)
            try:
                self.assertEqual(oriented.size, (40, 80))
            finally:
                oriented.close()

    def test_content_image_output_is_enabled(self):
        self.assertFalse(TEXT_ONLY_OUTPUT)

    def test_graphic_classifier_does_not_transcribe_content(self):
        from app.core.batch_ocr import classify_graphic_crop
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "crop.png"
            from PIL import Image
            Image.new("RGB", (1800, 1200), "white").save(crop)
            client = Mock()
            client.generate.return_value = SimpleNamespace(
                response='{"category":"signature","confidence":0.99}'
            )
            category, confidence = classify_graphic_crop(client, "model", crop)
        self.assertEqual((category, confidence), ("signature", 0.99))
        self.assertIn("không OCR", client.generate.call_args.kwargs["prompt"])
        self.assertEqual(client.generate.call_args.kwargs["options"]["num_ctx"], 8192)
        self.assertFalse(any(Path(directory).glob("*.classify_*.jpg")))

    def test_graphic_context_error_retries_once_with_smaller_crop(self):
        from app.core.batch_ocr import classify_graphic_crop
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "crop.png"
            from PIL import Image
            Image.new("RGB", (2000, 1600), "white").save(crop)
            client = Mock()
            client.generate.side_effect = [
                RuntimeError("exceed_context_size_error"),
                SimpleNamespace(response='{"category":"content_image","confidence":0.99}'),
            ]
            category, confidence = classify_graphic_crop(
                client, "model", crop, log_func=lambda _message: None,
            )
            sent_images = [call.kwargs["images"][0] for call in client.generate.call_args_list]
        self.assertEqual((category, confidence), ("content_image", 0.99))
        self.assertEqual(client.generate.call_count, 2)
        self.assertIn("classify_1024", sent_images[0])
        self.assertIn("classify_768", sent_images[1])

    def test_graphic_is_deleted_only_at_safe_category_confidence(self):
        from app.core.batch_ocr import _discard_classified_graphic
        self.assertTrue(_discard_classified_graphic("signature", 0.90))
        self.assertTrue(_discard_classified_graphic("stamp", 0.95))
        self.assertTrue(_discard_classified_graphic("text_fragment", 0.95))
        self.assertTrue(_discard_classified_graphic("page_canvas", 0.96))
        self.assertTrue(_discard_classified_graphic("decoration", 0.95))
        self.assertFalse(_discard_classified_graphic("signature", 0.89))
        self.assertFalse(_discard_classified_graphic("decoration", 0.94))
        self.assertFalse(_discard_classified_graphic("uncertain", 1.0))
        self.assertFalse(_discard_classified_graphic("content_image", 1.0))
        self.assertFalse(_discard_classified_graphic("logo", 1.0))

    def test_only_byte_identical_assets_are_deduplicated(self):
        from app.core.batch_ocr import deduplicate_exact_assets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images/a.png").write_bytes(b"same")
            (root / "images/b.png").write_bytes(b"same")
            markdown = "![A](images/a.png)\n\n![B](images/b.png)"
            result, paths, removed = deduplicate_exact_assets(
                markdown, ["images/a.png", "images/b.png"], root,
            )
        self.assertEqual(paths, ["images/a.png"])
        self.assertEqual(removed, 1)
        self.assertNotIn("images/b.png", result)
        self.assertEqual(result.count("!["), 1)

    def test_ollama_timeout_retries_twice_after_initial_attempt(self):
        client = Mock()
        client.generate.side_effect = TimeoutError("request timed out")
        with patch("app.core.batch_ocr.time.sleep"):
            with self.assertRaises(TimeoutError):
                generate_with_retry(client, {}, log_func=lambda _message: None)
        self.assertEqual(client.generate.call_count, 3)

    def test_all_generation_prompts_forbid_answering_exam_questions(self):
        combined = "\n".join((PROMPT, QUALITY_RETRY_INSTRUCTION, TABLE_STRUCTURE_REPAIR_PROMPT)).casefold()
        self.assertIn("never answer questions", combined)
        self.assertIn("không trả lời câu hỏi", combined)
        self.assertIn("visibly present", combined)

    def test_finalizer_removes_standalone_page_footer_artifacts(self):
        malformed = (
            "Nội dung chính\n2 | Thông tin tuyển sinh HUFLIT\n"
            "Trang 1/6 – Mã đề thi 001\nNội dung tiếp"
        )
        repaired, report = finalize_markdown(malformed, return_report=True)
        self.assertNotIn("Thông tin tuyển sinh HUFLIT", repaired)
        self.assertNotIn("Mã đề thi 001", repaired)
        self.assertEqual(report["page_artifacts_removed"], 2)

    def test_finalizer_splits_glued_headings(self):
        repaired = finalize_markdown("# Tiêu đề ## Phần hai ### Phần ba")
        self.assertEqual(repaired.splitlines(), ["# Tiêu đề", "## Phần hai", "### Phần ba"])

    def test_finalizer_splits_choices_and_removes_hfill(self):
        repaired = finalize_markdown(r"A. $x=1$ \hfill B. $x=2$ C. $x=3$ D. $x=4$")
        self.assertNotIn(r"\hfill", repaired)
        self.assertEqual(repaired.splitlines(), ["A. $x=1$", "B. $x=2$", "C. $x=3$", "D. $x=4$"])

    def test_choice_letters_inside_math_do_not_block_option_splitting(self):
        repaired = finalize_markdown(
            "A. ${}_{10}^{8}A$. B. ${}_{10}^{2}A.$\nC. ${}_{10}^{2}C$. D. $10^2.$"
        )
        self.assertEqual(len([line for line in repaired.splitlines() if re.match(r"^[A-D]\. ", line)]), 4)

    def test_choice_splitter_does_not_split_isolated_prose_abbreviation(self):
        prose = "Tác giả B. Trần trình bày nội dung trong báo cáo."
        self.assertEqual(finalize_markdown(prose), prose)

    def test_choices_inside_html_cell_use_breaks_not_newlines(self):
        repaired = finalize_markdown("<table><tr><td>A. một B. hai C. ba D. bốn</td></tr></table>")
        self.assertIn("A. một<br>B. hai<br>C. ba<br>D. bốn", repaired)

    def test_table_fragments_merge_across_page_comment_and_image(self):
        markdown = (
            "| Câu | Điểm |\n|---|---|\n| 1 | 0,5 |\n\n"
            "<!-- Page 2 -->\n\n![Chú thích](images/page2.png)\n\n"
            "| Câu | Điểm |\n|---|---|\n| 2 | 1,0 |"
        )
        repaired = finalize_markdown(markdown)
        self.assertEqual(repaired.count("| Câu | Điểm |"), 1)
        self.assertIn("images/page2.png", repaired)
        self.assertIn("| 2 | 1,0 |", repaired)

    def test_table_continuation_without_repeated_header_is_merged(self):
        markdown = (
            "| Câu | Điểm |\n|---|---|\n| 1 | 0,5 |\n\n"
            "<!-- Page 2 -->\n\n| 2 | 1,0 |\n| 3 | 1,5 |"
        )
        repaired = finalize_markdown(markdown)
        self.assertIn("| 3 | 1,5 |", repaired)
        self.assertLess(repaired.index("| 1 |"), repaired.index("| 2 |"))

    def test_numbered_table_continuation_merges_after_empty_trailing_column(self):
        markdown = (
            "| 11 | A | Đơn vị A | Đề tài A | GV A |  |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| 12 | B | Đơn vị B | Đề tài B | GV B |  |\n\n"
            "<!-- Page 4 -->\n\n"
            "| 13 | C | Đơn vị C | Đề tài C | GV C |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| 14 | D | Đơn vị D | Đề tài D | GV D |"
        )
        repaired = merge_markdown_tables(markdown)
        self.assertEqual(repaired.count("| :--- | :--- | :--- | :--- | :--- |"), 1)
        self.assertIn("| 12 | B | Đơn vị B | Đề tài B | GV B |", repaired)
        self.assertIn("| 13 | C | Đơn vị C | Đề tài C | GV C |", repaired)

    def test_lettered_table_continuation_is_not_number_specific(self):
        markdown = (
            "| Mục | Nội dung |\n| :--- | :--- |\n| A | Một |\n\n"
            "<!-- Page 2 -->\n\n| B | Hai |\n| :--- | :--- |\n| C | Ba |"
        )
        repaired = merge_markdown_tables(markdown)
        self.assertEqual(repaired.count("| :--- | :--- |"), 1)
        self.assertLess(repaired.index("| A |"), repaired.index("| B |"))

    def test_html_document_wrapper_does_not_split_markdown_table(self):
        markdown = (
            "| Mục | Nội dung |\n| :--- | :--- |\n| 12 | Trước |\n\n"
            "<!-- Page 2 -->\n<html>\n<body>\n\n"
            "| 13 | Sau |\n| :--- | :--- |\n| 14 | Cuối |\n\n</body>\n</html>"
        )
        repaired = merge_markdown_tables(markdown)
        self.assertEqual(repaired.count("| :--- | :--- |"), 1)
        self.assertNotIn("<html>", repaired.casefold())
        self.assertLess(repaired.index("| 12 |"), repaired.index("| 13 |"))

    def test_inline_html_wrapper_does_not_split_markdown_table(self):
        markdown = (
            "| Mục | Nội dung |\n| :--- | :--- |\n| 1 | Trước |\n\n"
            "<html><body>| 2 | Sau |\n| :--- | :--- |\n| 3 | Cuối |</body></html>"
        )
        repaired = merge_markdown_tables(markdown)
        self.assertNotRegex(repaired.casefold(), r"</?(?:html|body)\b")
        self.assertEqual(repaired.count("| :--- | :--- |"), 1)

    def test_doctype_wrapper_does_not_turn_continuation_row_into_header(self):
        markdown = (
            "| STT | Họ tên | Nội dung |\n"
            "| :--- | :--- | :--- |\n"
            "| 9 | Người A | Trước |\n\n"
            "<!-- Page 3 -->\n\n<!DOCTYPE html>\n\n"
            "| 10 | Người B | Sau |\n"
            "| :--- | :--- | :--- |\n"
            "| 11 | Người C | Cuối |"
        )
        repaired = merge_markdown_tables(markdown)
        self.assertNotIn("doctype", repaired.casefold())
        self.assertEqual(repaired.count("| :--- | :--- | :--- |"), 1)
        self.assertLess(repaired.index("| 9 |"), repaired.index("| 10 |"))
        self.assertLess(repaired.index("| 10 |"), repaired.index("| 11 |"))

    def test_html_tables_merge_across_page_figure(self):
        markdown = (
            "<table><tr><td>Một</td></tr></table>\n<!-- Page 2 -->\n"
            "![Ảnh](images/p2.png)\n"
            "<table><tr><td>Hai</td></tr></table>"
        )
        repaired = finalize_markdown(markdown)
        self.assertEqual(repaired.casefold().count("<table"), 1)
        self.assertIn("images/p2.png", repaired)

    def test_math_balance_stays_inside_table_cell_and_currency_is_untouched(self):
        repaired = finalize_markdown("| Công thức | Giá |\n|---|---|\n| $x=1 | $5 |")
        self.assertIn("| $x=1$ | $5 |", repaired)

    def test_html_table_delimiter_leaks_are_repaired(self):
        repaired = finalize_markdown("<table><tr><td>-$</td>$<td>$</td>$</tr></table>")
        self.assertIn("<td>$-$</td><td></td>", repaired)
        self.assertNotIn("</td>$<td>", repaired)

    def test_html_table_continuation_does_not_render_as_indented_code(self):
        markdown = (
            "<table>\n<tr><td>STT</td><td>Họ tên</td><td>Ghi chú</td></tr>\n"
            "\t<tr><td>12</td><td>A</td><td></td></tr>\n\n"
            "\t<tr>\n\t\t<td>13</td>\n\t\t<td>B</td>\n\t</tr>\n</table>"
        )
        repaired = finalize_markdown(markdown)
        self.assertIn("| 13 | B |  |", repaired)
        self.assertNotIn("<tr>", repaired)
        self.assertNotRegex(repaired, r"(?m)^[ \t]{4,}<td>")

    def test_markdown_table_merges_with_html_continuation_after_conversion(self):
        markdown = (
            "| STT | Họ tên | Đơn vị | Đề tài | Hướng dẫn | Ghi chú |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| 05 | A | X | Đề tài A | GV A |  |\n\n"
            "<!-- Page 3 -->\n\n"
            "<table><tr><td>06</td><td>B</td><td>Y</td><td>Đề tài B</td>"
            "<td>GV B</td><td></td></tr>"
            "<tr><td>07</td><td>C</td><td>Z</td><td>Đề tài C</td>"
            "<td>GV C</td><td></td></tr></table>"
        )
        repaired = finalize_markdown(markdown)
        self.assertEqual(repaired.count("| STT | Họ tên |"), 1)
        self.assertEqual(repaired.count("| :--- | :--- | :--- | :--- | :--- | :--- |"), 1)
        self.assertIn("| 06 | B | Y | Đề tài B | GV B |  |", repaired)
        self.assertIn("| 07 | C | Z | Đề tài C | GV C |  |", repaired)

    def test_magazine_footer_and_preceding_page_number_are_removed(self):
        markdown = (
            "Nội dung\n\n01\nThông tin Du lịch tháng 11/2023 #ttdl20231129\n\n"
            "Nội dung tiếp\nThông tin Du lịch tháng 6/2024 | #ttdl20240629 **13**\n"
            "04 **Thông tin Du lịch tháng 6/2024** #ttdl20240629"
        )
        repaired, report = finalize_markdown(markdown, return_report=True)
        self.assertNotIn("#ttdl", repaired)
        self.assertNotIn("\n01\n", f"\n{repaired}\n")
        self.assertEqual(report["page_artifacts_removed"], 4)

    def test_double_escaped_hfill_is_removed(self):
        repaired = finalize_markdown(r"A. $1$ \\hfill B. $2$")
        self.assertNotIn("hfill", repaired)
        self.assertEqual(repaired.splitlines(), ["A. $1$", "B. $2$"])

    def test_output_filename_preserves_and_normalizes_vietnamese(self):
        decomposed = Path("Ba\u0309n tie\u0302\u0301ng Vie\u0323\u0302t.pdf")
        self.assertEqual(output_markdown_path(Path("out"), decomposed).name, "Bản tiếng Việt.md")

    def test_heading_splitter_ignores_prose_code_and_table_hashes(self):
        markdown = "Văn bản dùng ## ký hiệu.\n\n```\n# not a heading ## still code\n```\n\n| Giá trị ## giữ nguyên |"
        repaired = finalize_markdown(markdown)
        self.assertIn("Văn bản dùng ## ký hiệu.", repaired)
        self.assertIn("# not a heading ## still code", repaired)
        self.assertIn("| Giá trị ## giữ nguyên |", repaired)

    def test_side_by_side_all_caps_heading_is_split_conservatively(self):
        repaired = finalize_markdown(
            "# BỘ GIÁO DỤC VÀ ĐÀO TẠO **KỲ THI TRUNG HỌC PHỔ THÔNG QUỐC GIA**"
        )
        self.assertEqual(
            repaired.splitlines(),
            ["# BỘ GIÁO DỤC VÀ ĐÀO TẠO", "# **KỲ THI TRUNG HỌC PHỔ THÔNG QUỐC GIA**"],
        )

    def test_prose_is_not_left_inside_math_delimiters(self):
        malformed = "$2109 đô la Mỹ, và giá trị HDI là 0,666 đứng thứ 116 trong số 188 nước năm 2015.$"
        cleaned = clean_markdown(malformed)
        self.assertNotIn("$", cleaned)
        self.assertIn("2109 đô la Mỹ", cleaned)

    def test_unclosed_math_marker_is_removed_from_prose(self):
        malformed = "$2109 đô la Mỹ, và giá trị HDI là 0,666 đứng thứ 116 trong số 188 nước"
        self.assertNotIn("$", clean_markdown(malformed))

    def test_blank_page_assistant_explanation_becomes_empty_markdown(self):
        response = (
            "The provided image appears to be a blank page with no visible text "
            "or content. Therefore, there is nothing to transcribe into Markdown "
            "format. If you have another image, please provide it."
        )
        self.assertEqual(clean_markdown(response), "")

    def test_qwen_empty_response_placeholder_becomes_empty_markdown(self):
        from app.core.batch_ocr import (
            BlankOCRResult, is_blank_ocr_response, ocr_qwen_images,
        )

        self.assertTrue(is_blank_ocr_response("(Empty response)"))
        self.assertEqual(clean_markdown("(Empty response)"), "")
        client = Mock()
        client.generate.return_value = [SimpleNamespace(response="(Empty response)")]
        result = ocr_qwen_images(client, MODEL, [Path("page.png")])
        self.assertIsInstance(result, BlankOCRResult)

    def test_qwen_no_content_placeholder_becomes_empty_markdown(self):
        from app.core.batch_ocr import is_blank_ocr_response

        self.assertTrue(is_blank_ocr_response("(no content)"))
        self.assertEqual(clean_markdown("(no content)"), "")

    def test_qwen_completely_blank_explanation_becomes_empty_markdown(self):
        response = (
            "The provided image is completely blank and contains no visible text, "
            "diagrams, tables, or other content. Therefore, according to the OCR-only "
            "rules specified, there is nothing to transcribe.\n\n"
            "**Output:**\n\n(no content)"
        )
        self.assertEqual(clean_markdown(response), "")

    def test_qwen_repeated_heading_ladder_is_treated_as_blank_failure(self):
        from app.core.batch_ocr import (
            BlankOCRResult, is_blank_ocr_response, ocr_qwen_images,
        )

        response = "\n\n".join(
            f"{'#' * level} 10/24/2023" for level in range(1, 6)
        )
        self.assertTrue(is_blank_ocr_response(response))
        self.assertEqual(clean_markdown(response), "")
        client = Mock()
        client.generate.return_value = [SimpleNamespace(response=response)]
        result = ocr_qwen_images(client, MODEL, [Path("page.png")])
        self.assertIsInstance(result, BlankOCRResult)

    def test_qwen_verbose_blank_instruction_echo_is_treated_as_blank_failure(self):
        from app.core.batch_ocr import (
            BlankOCRResult, is_blank_ocr_response, ocr_qwen_images,
        )

        response = (
            "The image provided is a scanned document page that appears to be mostly "
            "blank, with very faint and possibly illegible text at the bottom. There "
            "are two red circular stamps visible near the bottom center of the page. "
            "One stamp contains the word 10HAY in reverse, and the other has indistinct "
            "markings. Below these stamps, there is extremely faint text that is too "
            "blurry to transcribe accurately. "
            "Per your instructions: If the page contains no visible document content, "
            "return an empty response. Do not describe the blank page and do not explain "
            "that there is nothing to transcribe. Given that the only potentially "
            "readable elements are minimal, reversed, or ambiguous and do not constitute "
            "clear transcribable document content under the stated OCR rules, I will "
            "return an empty response as required."
        )
        self.assertGreater(len(response.split()), 120)
        self.assertTrue(is_blank_ocr_response(response))
        self.assertEqual(clean_markdown(response), "")
        client = Mock()
        client.generate.return_value = [SimpleNamespace(response=response)]
        result = ocr_qwen_images(client, MODEL, [Path("page.png")])
        self.assertIsInstance(result, BlankOCRResult)

    def test_document_sentence_about_blank_page_is_not_removed(self):
        document = (
            "# Hướng dẫn\n\nTrang trắng được dùng để phân cách các phụ lục. "
            "Nội dung này phải được giữ nguyên trong tài liệu."
        )
        self.assertEqual(clean_markdown(document), document)

    def test_qwen_blank_response_has_explicit_empty_result_state(self):
        from app.core.batch_ocr import BlankOCRResult, ocr_qwen_images

        client = Mock()
        client.generate.return_value = [SimpleNamespace(response=(
            "The provided image appears to be a blank page with no visible text. "
            "There is nothing to transcribe into Markdown format."
        ))]
        result = ocr_qwen_images(client, MODEL, [Path("page.png")])
        self.assertIsInstance(result, BlankOCRResult)
        self.assertEqual(result, "")

    def test_low_diacritic_vietnamese_requests_vision_retry(self):
        malformed = "Viet Nam la nuoc co muc thu nhap trung binh va chuong trinh trong nam. " * 12
        self.assertTrue(needs_vision_retry(malformed))

    def test_normal_vietnamese_does_not_request_vision_retry(self):
        self.assertFalse(needs_vision_retry("Nội dung tiếng Việt hợp lệ và có đầy đủ dấu câu."))

    def test_mixed_language_retry_uses_specialized_instruction(self):
        text = (
            "This is the system API and the report is for users. "
            "Viet Nam la mot nuoc co nhieu diem du lich va cac chuong trinh phat trien. "
        ) * 3
        instruction = quality_retry_instruction(text, "missing_vietnamese_diacritics")
        self.assertIn("mixed Vietnamese-English", instruction)
        self.assertIn("without adding Vietnamese diacritics", instruction)

    def test_repetition_retry_uses_loop_specific_instruction(self):
        instruction = quality_retry_instruction("Nội dung", "repetition_loop")
        self.assertIn("entered a repetition loop", instruction)
        self.assertIn("physical bottom of the image", instruction)

    def test_inconsistent_html_colspan_requests_table_retry(self):
        table = '<table><tr><th colspan="2">A</th></tr><tr><td>B</td></tr></table>'
        self.assertTrue(needs_table_retry(table))

    def test_complex_html_table_is_never_converted_to_pipe_table(self):
        table = (
            '<table><tr><th rowspan="2">Ngành</th><th colspan="2">Chỉ tiêu</th></tr>'
            '<tr><td>2025</td><td>2026</td></tr></table>'
        )
        repaired = finalize_markdown(table)
        self.assertIn('rowspan="2"', repaired)
        self.assertIn('colspan="2"', repaired)
        self.assertNotIn("| Ngành |", repaired)

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

    def test_html_conversion_escapes_cell_content(self):
        markdown = "| A | B |\n|---|---|\n| <script> | 1 | extra |"
        repaired = repair_markdown_tables(markdown)
        self.assertIn("&lt;script&gt;", repaired)
        self.assertNotIn("<script>", repaired)

    def test_separator_column_mismatch_becomes_html(self):
        markdown = "| A | B |\n|---|\n| 1 | 2 |"
        repaired = repair_markdown_tables(markdown)
        self.assertIn("<th>A</th>", repaired)
        self.assertIn("<td>2</td>", repaired)

    def test_pipe_inside_code_span_does_not_create_extra_cell(self):
        markdown = "| Cú pháp | Mô tả |\n|---|---|\n| `a|b` | lựa chọn |"
        self.assertEqual(repair_markdown_tables(markdown), markdown)

    def test_merge_markdown_tables_with_pseudo_header_continuation(self):
        table_page_1 = (
            "| STT | Họ và tên | Đơn vị |\n"
            "| :--- | :--- | :--- |\n"
            "| 1 | Nguyễn Văn A | Hà Nội |\n"
            "| 2 | Trần Thị B | Đà Nẵng |"
        )
        table_page_2 = (
            "| 3 | Lê Văn C | TP.HCM |\n"
            "| :--- | :--- | :--- |\n"
            "| 4 | Phạm Thị D | Cần Thơ |"
        )
        content = f"{table_page_1}\n\n<!-- Page 2 -->\n\n{table_page_2}"
        merged = merge_markdown_tables(content)
        self.assertEqual(content.count("| :--- | :--- | :--- |"), 2)
        self.assertEqual(merged.count("| :--- | :--- | :--- |"), 1)
        self.assertIn("| 1 | Nguyễn Văn A | Hà Nội |", merged)
        self.assertIn("| 2 | Trần Thị B | Đà Nẵng |", merged)
        self.assertIn("| 3 | Lê Văn C | TP.HCM |", merged)
        self.assertIn("| 4 | Phạm Thị D | Cần Thơ |", merged)

    def test_merge_markdown_tables_hierarchical_stt_continuation(self):
        page_1 = (
            "| STT | Chỉ tiêu | ĐVT | Số liệu |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| 4.1. | Nhiệm vụ giao | Vụ | 10 |\n"
            "| 4.2. | Đúng hạn | Vụ | 8 |"
        )
        page_2 = (
            "| STT | Chỉ tiêu | ĐVT | Số liệu |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| 4.3. | Quá hạn | Vụ | 2 |\n"
            "| 5. | Khảo sát | Phiếu | 100 |"
        )
        content = f"{page_1}\n\n<!-- Page 2 -->\n\n{page_2}"
        merged = merge_markdown_tables(content)
        self.assertEqual(merged.count("| :--- | :--- | :--- | :--- |"), 1)
        self.assertIn("| 4.2. | Đúng hạn | Vụ | 8 |", merged)
        self.assertIn("| 4.3. | Quá hạn | Vụ | 2 |", merged)
        self.assertIn("| 5. | Khảo sát | Phiếu | 100 |", merged)

    def test_merge_html_tables_deduplicates_repeated_header(self):
        page_1 = (
            "<table>\n"
            "<thead><tr><th>STT</th><th>Nội dung</th></tr></thead>\n"
            "<tbody><tr><td>1</td><td>Dòng 1</td></tr></tbody>\n"
            "</table>"
        )
        page_2 = (
            "<table>\n"
            "<thead><tr><th>STT</th><th>Nội dung</th></tr></thead>\n"
            "<tbody><tr><td>2</td><td>Dòng 2</td></tr></tbody>\n"
            "</table>"
        )
        content = f"{page_1}\n\n<!-- Page 2 -->\n\n{page_2}"
        merged = merge_markdown_tables(content)
        self.assertEqual(merged.count("<table>"), 1)
        self.assertEqual(merged.count("</table>"), 1)
        self.assertEqual(merged.count("<th>STT</th>"), 1)
        self.assertIn("<td>1</td>", merged)
        self.assertIn("<td>2</td>", merged)

    def test_finalizer_normalizes_heading_and_paragraph_breaks(self):
        malformed = "#Tiêu đề\n\n####Mục con\n\nĐây là một câu bị\nngắt giữa dòng."
        repaired = finalize_markdown(malformed)
        self.assertIn("# Tiêu đề", repaired)
        self.assertIn("## Mục con", repaired)
        self.assertIn("Đây là một câu bị ngắt giữa dòng.", repaired)

    def test_finalizer_joins_broken_list_item(self):
        malformed = "- Nội dung thứ nhất\n  tiếp tục ở dòng sau\n- Nội dung thứ hai"
        repaired = finalize_markdown(malformed)
        self.assertIn("- Nội dung thứ nhất tiếp tục ở dòng sau", repaired)

    def test_finalizer_closes_html_table_tags(self):
        malformed = "<table>\n<tr>\n<td>Nội dung"
        repaired = finalize_markdown(malformed)
        self.assertIn("</td></tr></table>", repaired)

    def test_finalizer_normalizes_and_deduplicates_image_paths(self):
        malformed = "![Ảnh](.\\images\\one.png)\n\n![Bản sao](images/one.png)"
        repaired = finalize_markdown(malformed)
        self.assertEqual(repaired.count("images/one.png"), 1)

    def test_finalizer_reports_each_repair_category(self):
        malformed = (
            "#Tiêu đề\n\nĐây là câu\nbị ngắt.\n\n"
            "- Mục\n  tiếp tục\n\n<table><tr><td>Ô\n\n"
            "![Ảnh](.\\images\\one.png)\n![Lặp](images/one.png)"
        )
        _, report = finalize_markdown(malformed, return_report=True)
        self.assertEqual(report["headings"], 1)
        self.assertEqual(report["paragraph_lines_joined"], 1)
        self.assertEqual(report["list_lines_joined"], 1)
        self.assertEqual(report["html_tags_closed"], 3)
        self.assertEqual(report["image_paths"], 1)
        self.assertEqual(report["duplicate_images"], 1)

    def test_finalizer_warns_without_changing_spelling_by_default(self):
        repaired, report = finalize_markdown("Kết quả nghien cuu đã hoàn thành.", return_report=True)
        self.assertIn("nghien cuu", repaired)
        self.assertEqual(report["spell_fixed"], 0)
        self.assertEqual(report["spell_warnings"], 2)
        self.assertEqual(report["spelling_warnings"][0]["line"], 1)
        self.assertEqual(report["spelling_warnings"][0]["confidence"], "high")

    def test_finalizer_only_corrects_spelling_when_opted_in(self):
        repaired, report = finalize_markdown(
            "Kết quả nghien cuu đã hoàn thành.", return_report=True, spell_correct=True
        )
        self.assertIn("nghiên cứu", repaired)
        self.assertEqual(report["spell_fixed"], 2)

    def test_finalizer_collapses_long_repetition_loop(self):
        line = "Đây là một đoạn văn dài bị mô hình lặp lại liên tục do ảnh nền phức tạp."
        repaired, report = finalize_markdown("\n".join([line] * 40), return_report=True)
        self.assertEqual(repaired.count(line), 1)
        self.assertEqual(report["repetition_lines_removed"], 39)

    def test_finalizer_collapses_repeated_multiline_block(self):
        block = [
            "Nội dung thứ nhất của một chuỗi văn bản đủ dài để kiểm tra.",
            "Nội dung thứ hai của một chuỗi văn bản đủ dài để kiểm tra.",
        ]
        repaired = finalize_markdown("\n".join(block * 5))
        self.assertEqual(repaired.count("Nội dung thứ nhất"), 1)
        self.assertEqual(repaired.count("Nội dung thứ hai"), 1)

    def test_repetition_guard_preserves_code_math_and_tables(self):
        line = "Dòng có chủ ý được lặp lại nhiều lần trong vùng được bảo vệ."
        markdown = (
            "```text\n" + "\n".join([line] * 3) + "\n```\n"
            "$$\n" + "\n".join([line] * 3) + "\n$$\n"
            "<table><tr><td>" + line + "</td></tr></table>"
        )
        repaired = finalize_markdown(markdown)
        self.assertEqual(repaired.count(line), 7)

    def test_cross_page_paragraph_stitching(self):
        markdown = (
            "Trong 5 tháng đầu năm 2007 xã đã triển khai thu quỹ An ninh trật tự, tổ chức huấn\n\n"
            "<!-- Page 2 -->\n\n"
            "luyện dân quân 2007 đạt loại khá. Đăng ký tuổi 17 đạt 100%."
        )
        repaired = finalize_markdown(markdown)
        self.assertIn("tổ chức huấn <!-- Page 2 --> luyện dân quân 2007", repaired)


    def test_generic_publication_footer_only_removed_near_page_end(self):
        markdown = (
            "<!-- Page 1 -->\n#topic20231129\nNội dung chính dòng một.\n"
            "Nội dung chính dòng hai.\nNội dung chính dòng ba.\nNội dung chính dòng bốn.\n"
            "Dòng nội dung cuối cùng của trang đủ dài để được giữ nguyên.\n#issue20240101\n"
            "<!-- Page 2 -->\nNội dung trang hai.\nISSN 1234-5678"
        )
        repaired = finalize_markdown(markdown)
        self.assertIn("topic20231129", repaired)
        self.assertNotIn("#issue20240101", repaired)
        self.assertNotIn("# issue20240101", repaired)
        self.assertNotIn("ISSN 1234-5678", repaired)

    def test_standalone_page_number_footer_removed(self):
        markdown = (
            "<!-- Page 1 -->\n"
            "Nội dung dòng một của trang.\n"
            "Nội dung dòng hai của trang.\n"
            "- 1 -\n\n"
            "<!-- Page 2 -->\n"
            "Nội dung trang hai.\n"
            "[ 2 ]"
        )
        repaired = finalize_markdown(markdown)
        self.assertNotIn("- 1 -", repaired)
        self.assertNotIn("[ 2 ]", repaired)

    def test_rotated_page_footer_number_is_removed_at_start_of_page_block(self):
        markdown = (
            "<!-- Page 2 -->\n| Stt | Họ và tên |\n| --- | --- |\n| 12 | A |\n\n"
            "<!-- Page 3 -->\n\n2\n\n"
            "| Stt | Họ và tên |\n| --- | --- |\n| 13 | B |"
        )
        repaired, report = finalize_markdown(markdown, return_report=True)
        self.assertNotIn("\n2\n", f"\n{repaired}\n")
        self.assertIn("| 13 | B |", repaired)
        self.assertEqual(report["page_artifacts_removed"], 1)




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

    def test_finalizer_removes_all_orphan_image_placeholder_forms(self):
        markdown = (
            "Trước\n\n![Thiếu](images/image_placeholder.png)\n\n"
            '<img alt="Thiếu" src="image_placeholder.jpg">\n\n'
            "image_placeholder\n\n![Giữ](images/real.png)\n\nSau"
        )
        repaired = finalize_markdown(markdown)
        self.assertNotIn("image_placeholder", repaired.casefold())
        self.assertNotIn("<img", repaired.casefold())
        self.assertIn("![Giữ](images/real.png)", repaired)
        self.assertIn("Trước", repaired)
        self.assertIn("Sau", repaired)

    def test_page_asset_cleanup_uses_path_aware_placeholder_pattern(self):
        from app.core.batch_ocr import apply_page_assets

        repaired = apply_page_assets(
            "![Image](images/image_placeholder.png)\n\n"
            "![App](./image_placeholder)\n\nNội dung",
            1, [],
        )
        self.assertNotIn("image_placeholder", repaired.casefold())
        self.assertEqual(repaired, "Nội dung")

    def test_unplaced_assets_are_preserved_at_page_end(self):
        from app.core.batch_ocr import apply_page_assets

        result = apply_page_assets(
            "Nội dung trang", 1,
            ["images/report_page_1_draw_2.png", "images/report_page_1_img_1.png"],
        )
        self.assertIn("draw_2.png", result)
        self.assertIn("img_1.png", result)

    def test_cleanup_removes_only_unreferenced_local_assets(self):
        from app.core.batch_ocr import cleanup_unreferenced_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            kept = root / "images" / "kept.png"
            removed = root / "images" / "removed.png"
            kept.write_bytes(b"image")
            removed.write_bytes(b"image")
            count = cleanup_unreferenced_assets(
                "![Hình](images/kept.png)",
                ["images/kept.png", "images/removed.png", "../outside.png"],
                root,
            )
            self.assertEqual(count, 1)
            self.assertTrue(kept.exists())
            self.assertFalse(removed.exists())

    def test_explicit_placeholder_can_still_use_vector_crop(self):
        from app.core.batch_ocr import apply_page_assets

        result = apply_page_assets(
            "![Sơ đồ quy trình](image_placeholder.png)", 1,
            ["images/report_page_1_draw_2.png"],
        )
        self.assertIn("draw_2.png", result)

    def test_signature_and_stamp_alt_text_removes_link_after_asset_placement(self):
        from app.core.batch_ocr import apply_page_assets

        result = apply_page_assets(
            "Nội dung\n\n![Seal and Signature](image_placeholder.png)",
            1, ["images/document_page_1_layout_img_1.png"],
        )
        self.assertNotIn("layout_img_1.png", result)
        self.assertNotIn("Seal and Signature", result)
        self.assertIn("Nội dung", result)

    def test_logo_alt_text_is_not_removed(self):
        from app.core.batch_ocr import apply_page_assets

        result = apply_page_assets(
            "![University Logo](image_placeholder.png)",
            1, ["images/document_page_1_layout_img_1.png"],
        )
        self.assertIn("University Logo", result)
        self.assertIn("layout_img_1.png", result)

    def test_unplaced_bitmaps_are_preserved_without_layout_blocks(self):
        from app.core.batch_ocr import apply_page_assets

        images = ["images/top_img_1.png", "images/bottom_img_2.png"]
        markdown = "Đoạn văn đầu tiên của trang.\n\nĐoạn văn ở giữa trang.\n\nĐoạn văn cuối trang."
        result = apply_page_assets(markdown, 1, images)
        self.assertIn("top_img_1.png", result)
        self.assertIn("bottom_img_2.png", result)

    def test_coordinate_fallback_does_not_insert_inside_table(self):
        from app.core.batch_ocr import apply_page_assets

        images = ["images/figure_img_1.png"]
        markdown = "Đoạn mở đầu.\n\n<table><tr><td>Dữ liệu</td></tr></table>\n\nĐoạn kết thúc."
        result = apply_page_assets(markdown, 1, images)
        self.assertIn("figure_img_1.png", result)
        self.assertGreater(result.index("figure_img_1.png"), result.index("</table>"))

    def test_vector_line_and_margin_filter(self):
        from app.core.batch_ocr import _keep_vector_drawing
        from unittest.mock import MagicMock

        def rect(x0, y0, x1, y1):
            value = MagicMock()
            value.is_empty = False
            value.x0, value.y0, value.x1, value.y1 = x0, y0, x1, y1
            value.width, value.height = x1 - x0, y1 - y0
            return value

        self.assertFalse(_keep_vector_drawing(rect(10, 100, 500, 105), 600, 800))
        self.assertFalse(_keep_vector_drawing(rect(10, 100, 500, 120), 600, 800))
        self.assertFalse(_keep_vector_drawing(rect(100, 10, 200, 80), 600, 800))
        self.assertFalse(_keep_vector_drawing(rect(100, 750, 200, 790), 600, 800))
        self.assertTrue(_keep_vector_drawing(rect(100, 100, 240, 180), 600, 800))

    def test_full_page_canvas_filter_uses_eighty_percent_threshold(self):
        # pyrefly: ignore [missing-import]
        import pymupdf
        from app.core.batch_ocr import _is_full_page_canvas

        self.assertTrue(_is_full_page_canvas(pymupdf.Rect(0, 0, 600, 760), 600, 800))
        self.assertFalse(_is_full_page_canvas(pymupdf.Rect(0, 0, 300, 400), 600, 800))

    def test_uncaptioned_graphic_requires_classification_at_any_position(self):
        # pyrefly: ignore [missing-import]
        import pymupdf
        from app.core.batch_ocr import _needs_graphic_classification

        page = SimpleNamespace(rect=pymupdf.Rect(0, 0, 600, 800))
        page.get_text = lambda _kind: []
        self.assertTrue(_needs_graphic_classification(page, pymupdf.Rect(360, 650, 520, 780)))
        self.assertTrue(_needs_graphic_classification(page, pymupdf.Rect(60, 50, 220, 180)))

        page.get_text = lambda _kind: [(350, 690, 520, 715, "Hình", 0, 0, 0), (350, 715, 520, 740, "2", 0, 0, 1)]
        self.assertFalse(_needs_graphic_classification(page, pymupdf.Rect(360, 650, 520, 780)))

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
                "<!-- Page 1 -->\n\n# Result\n\n![Hình ảnh](images/figure.png)",
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

    def test_text_heavy_layout_region_is_not_extracted_as_image(self):
        from app.core.batch_ocr import extract_images_from_page
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "source.pdf"
            page_image_path = root / "page.png"
            # pyrefly: ignore [missing-import]
            from PIL import Image
            Image.new("RGB", (1000, 1000), "white").save(page_image_path)

            detected_images = [(100.0, 100.0, 500.0, 300.0)]
            detector = MagicMock()
            page = MagicMock()
            page.rect = MagicMock(width=500, height=500)
            page.get_text.return_value = [
                (55, 55, 95, 70, f"word{index}", 0, 0, index)
                for index in range(12)
            ]
            page.get_images.return_value = []
            page.get_drawings.return_value = []
            document = MagicMock()
            document.load_page.return_value = page

            with patch("pymupdf.open", return_value=document):
                result = extract_images_from_page(
                    pdf_path, 0, root / "images", "sample",
                    layout_detector=detector,
                    page_image_path=page_image_path,
                    detected_images=detected_images,
                )

            self.assertEqual(result, [])
            self.assertEqual(detected_images, [])

    def test_layout_image_overlapping_table_is_not_extracted(self):
        from app.core.batch_ocr import extract_images_from_page
        from unittest.mock import MagicMock, patch
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_image = root / "page.png"
            Image.new("RGB", (1000, 1000), "white").save(page_image)
            images = [(100.0, 100.0, 900.0, 800.0)]
            page = MagicMock()
            page.rect = MagicMock(width=500, height=500)
            page.get_images.return_value = []
            page.get_drawings.return_value = []
            document = MagicMock()
            document.load_page.return_value = page
            with patch("pymupdf.open", return_value=document):
                result = extract_images_from_page(
                    root / "source.pdf", 0, root / "images", "sample",
                    layout_detector=MagicMock(), page_image_path=page_image,
                    detected_images=images,
                    table_regions=[(80.0, 80.0, 920.0, 820.0)],
                )
        self.assertEqual(result, [])
        self.assertEqual(images, [])

    def test_image_crop_padding_is_proportional_and_clamped(self):
        from app.core.batch_ocr import _padded_box

        self.assertEqual(_padded_box((5, 8, 105, 208), 1000, 1000), (0.0, 0.0, 115.0, 218.0))
        self.assertEqual(_padded_box((900, 900, 995, 995), 1000, 1000), (890.0, 890.0, 1000, 1000))

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
        from app.core.layout_detector import PageLayoutAnalysis
        formula_boxes = [(50.0, 100.0, 150.0, 150.0), (200.0, 200.0, 300.0, 250.0)]
        mock_detector.analyse_page.return_value = PageLayoutAnalysis(
            segments=[],
            blocks=[("formula", box) for box in formula_boxes],
            tables=[], images=[], formulas=formula_boxes, regions=formula_boxes,
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
            # pyrefly: ignore [missing-import]
            from PIL import Image
            img = Image.new("RGB", (1000, 1000), "white")
            img_path = Path(temp_dir) / "page_1.png"
            img.save(img_path)

            pdf_path = Path(temp_dir) / "input.pdf"
            pdf_path.write_bytes(b"pdf")

            output_dir = Path(temp_dir) / "output"
            result = process_single_pdf(
                pdf_path, output_dir, client, "qwen3.5:4b",
                skip_blank_pages=False,
            )

            # Read output markdown
            md_content = result.read_text(encoding="utf-8")
            self.assertIn("$E = mc^2$", md_content)
            self.assertIn("$a^2 + b^2 = c^2$", md_content)

    def test_merge_html_tables_deduplicates_multi_tier_repeated_headers(self):
        page_1 = (
            "<table>\n"
            "<tr><th rowspan=\"2\">STT</th><th colspan=\"2\">Kết quả</th></tr>\n"
            "<tr><th>ĐVT</th><th>Số liệu</th></tr>\n"
            "<tr><td>1</td><td>Văn bản</td><td>10</td></tr>\n"
            "</table>"
        )
        page_2 = (
            "<table>\n"
            "<tr><th rowspan=\"2\">STT</th><th colspan=\"2\">Kết quả</th></tr>\n"
            "<tr><th>ĐVT</th><th>Số liệu</th></tr>\n"
            "<tr><td>2</td><td>Văn bản</td><td>20</td></tr>\n"
            "</table>"
        )
        content = f"{page_1}\n\n<!-- Page 2 -->\n\n{page_2}"
        merged = merge_markdown_tables(content)
        self.assertEqual(merged.count("<table>"), 1)
        self.assertEqual(merged.count("</table>"), 1)
        self.assertEqual(merged.count("<th>ĐVT</th>"), 1)
        self.assertIn("<td>1</td>", merged)
        self.assertIn("<td>2</td>", merged)

    def test_clean_html_table_intermediate_headers(self):
        content = (
            "<table>\n"
            "<tr><th>STT</th><th>Nội dung</th><th>Số liệu</th></tr>\n"
            "<tr><td>1</td><td>Chỉ tiêu 1</td><td>100</td></tr>\n"
            "<tr><th>Đơn vị tính</th><th>Số liệu</th></tr>\n"
            "<tr><td>2</td><td>Chỉ tiêu 2</td><td>200</td></tr>\n"
            "</table>"
        )
        merged = merge_markdown_tables(content)
        self.assertNotIn("<tr><th>Đơn vị tính</th><th>Số liệu</th></tr>", merged)
        self.assertIn("<td>1</td>", merged)
        self.assertIn("<td>2</td>", merged)


if __name__ == "__main__":
    unittest.main()
