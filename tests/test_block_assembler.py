import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.block_assembler import DocumentBlock, assemble_blocks
from app.core.batch_ocr import apply_page_assets, ocr_coordinate_blocks


class BlockAssemblerTests(unittest.TestCase):
    def test_image_is_inserted_between_coordinate_blocks(self):
        heading = DocumentBlock("heading", (0, 0, 800, 80))
        before = DocumentBlock("text", (0, 100, 800, 250))
        image = DocumentBlock("image", (100, 270, 700, 500))
        after = DocumentBlock("text", (0, 520, 800, 700))
        markdown = assemble_blocks(
            [heading, before, image, after],
            {
                heading.bbox: "# Tiêu đề",
                before.bbox: "Đoạn trước hình.",
                after.bbox: "Đoạn sau hình.",
            },
            ["images/figure.png"],
        )
        self.assertLess(markdown.index("Đoạn trước"), markdown.index("images/figure.png"))
        self.assertLess(markdown.index("images/figure.png"), markdown.index("Đoạn sau"))

    def test_unmatched_images_are_preserved(self):
        block = DocumentBlock("text", (0, 0, 100, 100))
        markdown = assemble_blocks([block], {block.bbox: "Text"}, ["images/extra.png"])
        self.assertIn("images/extra.png", markdown)

    def test_coordinate_ocr_interleaves_image_without_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "page.png"
            from PIL import Image
            Image.new("RGB", (800, 700), "white").save(page)
            blocks = [
                ("text", (0, 0, 800, 200)),
                ("image", (100, 220, 700, 450)),
                ("text", (0, 470, 800, 700)),
            ]
            client = Mock()
            client.generate.return_value = [SimpleNamespace(
                response="Đoạn trước\n\n![Hình](image_placeholder.png)\n\nĐoạn sau"
            )]
            markdown = ocr_coordinate_blocks(
                client, "model", page, blocks, ["images/figure.png"], root / "blocks"
            )
        self.assertEqual(client.generate.call_count, 1)
        self.assertLess(markdown.index("Đoạn trước"), markdown.index("images/figure.png"))
        self.assertLess(markdown.index("images/figure.png"), markdown.index("Đoạn sau"))

    def test_already_positioned_image_is_not_appended_twice(self):
        markdown = "Trước\n\n![Hình](images/figure.png)\n\nSau"
        result = apply_page_assets(markdown, 1, ["images/figure.png"])
        self.assertEqual(result.count("images/figure.png"), 1)

    def test_surplus_image_placeholder_is_removed_without_missing_reference(self):
        markdown = (
            "Trước\n\n![Hình 1](image_placeholder.png)\n\n"
            "Giữa\n\n![Nhận nhầm công thức](image_placeholder.png)\n\nSau"
        )
        result = apply_page_assets(markdown, 1, ["images/figure.png"])
        self.assertEqual(result.count("images/figure.png"), 1)
        self.assertNotIn("image_placeholder", result)

    def test_placeholder_variants_are_resolved_in_reading_order(self):
        markdown = (
            "![Một](images/image_placeholder.png)\n\n"
            '<img alt="Hai" src="image_placeholder.jpg">\n\n'
            "image_placeholder"
        )
        result = apply_page_assets(
            markdown, 2,
            ["images/one.png", "images/two.png", "images/three.png"],
        )
        self.assertLess(result.index("images/one.png"), result.index("images/two.png"))
        self.assertLess(result.index("images/two.png"), result.index("images/three.png"))
        self.assertNotIn("image_placeholder", result)

    def test_real_image_and_remote_url_are_not_replaced(self):
        markdown = (
            "![Đã gắn](images/existing.png)\n\n"
            "![Ảnh mạng](https://example.com/photo.png)"
        )
        result = apply_page_assets(markdown, 1, ["images/new.png"])
        self.assertIn("images/existing.png", result)
        self.assertIn("https://example.com/photo.png", result)
        self.assertEqual(result.count("images/new.png"), 1)


if __name__ == "__main__":
    unittest.main()
