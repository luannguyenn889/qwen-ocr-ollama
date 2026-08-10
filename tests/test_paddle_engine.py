import unittest

from app.core.paddle_engine import PaddleOCREngine


class FakePaddleOCR:
    def predict(self, image_path):
        return [{
            "rec_texts": ["dòng dưới", "dòng trên", "nhiễu"],
            "rec_scores": [0.99, 0.98, 0.2],
            "rec_polys": [
                [[10, 40], [20, 40], [20, 50], [10, 50]],
                [[10, 10], [20, 10], [20, 20], [10, 20]],
                [[10, 60], [20, 60], [20, 70], [10, 70]],
            ],
        }]


class PaddleEngineTests(unittest.TestCase):
    def test_predict_results_are_sorted_and_low_confidence_text_is_dropped(self):
        engine = PaddleOCREngine(confidence_threshold=0.5)
        engine._ocr = FakePaddleOCR()
        self.assertEqual(engine.ocr_image("samples/images/01_scan_ro.png"), "dòng trên\ndòng dưới")


if __name__ == "__main__":
    unittest.main()
