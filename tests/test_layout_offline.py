import os
import unittest
from unittest.mock import patch

from app.core.layout_detector import create_layout_detector
from app.core.formula_ocr import formula_ocr_status, recognize_formula


class LayoutOfflineTests(unittest.TestCase):
    def test_layout_can_be_explicitly_disabled_without_loading_models(self):
        with patch.dict(os.environ, {"QWEN_OCR_DISABLE_LAYOUT": "1"}):
            detector, status = create_layout_detector()
        self.assertIsNone(detector)
        self.assertIn("disabled", status)

    def test_formula_model_unavailable_is_non_fatal(self):
        with patch("app.core.formula_ocr.get_formula_model", return_value=False):
            self.assertIsNone(recognize_formula(b"not-an-image"))

    def test_missing_pix2tex_has_explicit_fallback_status(self):
        with (
            patch("app.core.formula_ocr.importlib.util.find_spec", return_value=None),
            patch("app.core.formula_ocr._model_instance", None),
        ):
            available, status = formula_ocr_status()
        self.assertFalse(available)
        self.assertIn("Qwen fallback", status)


if __name__ == "__main__":
    unittest.main()
