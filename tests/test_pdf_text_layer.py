import tempfile
import unittest
from pathlib import Path

import pymupdf

from app.core.pdf_ocr import ocr_pdf
from app.core.pdf_text_layer import extract_native_text


class NeverCalledEngine:
    def check_connection(self):
        raise AssertionError("PDF có text layer không được kết nối Ollama")

    def ocr_image(self, image_path):
        raise AssertionError("PDF có text layer không được OCR thành ảnh")


class PdfTextLayerTests(unittest.TestCase):
    def test_native_text_is_preferred_to_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "native.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Native PDF text must not be sent to vision OCR.")
            document.save(pdf_path)
            document.close()

            document = pymupdf.open(pdf_path)
            native = extract_native_text(document[0])
            document.close()
            self.assertTrue(native.is_usable)

            output_path = ocr_pdf(NeverCalledEngine(), pdf_path, root / "out")
            output = output_path.read_text(encoding="utf-8")
            self.assertIn("nguồn: native-text", output)
            self.assertIn("Native PDF text", output)
            self.assertNotIn("![", output)


if __name__ == "__main__":
    unittest.main()
