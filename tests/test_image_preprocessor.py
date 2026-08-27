import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from app.core.image_preprocessor import is_scan_degraded, enhance_scanned_document


class ImagePreprocessorTests(unittest.TestCase):
    def test_detect_degraded_yellow_document(self):
        # Create a synthetic yellowish background with black text
        img = np.full((300, 300, 3), (120, 200, 240), dtype=np.uint8)  # Yellowish BGR (Low Blue, High Red/Green)
        cv2.putText(img, "ĐỘC LẬP", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

        self.assertTrue(is_scan_degraded(img))

        # Enhance it
        enhanced = enhance_scanned_document(img)
        self.assertIsNotNone(enhanced)
        # Low resolution (300x300) should be upscaled 3x to 900x900
        self.assertEqual(enhanced.shape, (900, 900, 3))

        # Background should be significantly brighter/whiter
        bg_pixel = enhanced[10, 10]
        self.assertGreater(int(bg_pixel[0]), 220)

    def test_clean_document_skipped_or_enhanced_safely(self):
        # Create a clean white background with black text
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.putText(img, "VAN BAN PHAP LUAT", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        self.assertFalse(is_scan_degraded(img))

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "enhanced.png"
            result = enhance_scanned_document(img, output_path=out_file)
            self.assertTrue(out_file.is_file())
            self.assertEqual(result.shape, img.shape)

    def test_enhance_scanned_document_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input.png"
            out_file = Path(tmp_dir) / "output.png"

            # Create noisy image
            img = np.full((200, 200, 3), 200, dtype=np.uint8)
            cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
            cv2.imwrite(str(in_file), img)

            enhanced = enhance_scanned_document(in_file, output_path=out_file, force=True)
            self.assertTrue(out_file.is_file())
            # 200x200 should be upscaled 3x to 600x600
            self.assertEqual(enhanced.shape, (600, 600, 3))


if __name__ == "__main__":
    unittest.main()
