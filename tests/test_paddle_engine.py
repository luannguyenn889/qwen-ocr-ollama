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

    def test_small_y_jitter_is_clustered_before_x_sort(self):
        engine = PaddleOCREngine()
        engine._ocr = type("JitteredOCR", (), {"predict": lambda self, _path: [{
            "rec_texts": ["bên phải", "bên trái", "dòng sau"],
            "rec_scores": [0.99, 0.99, 0.99],
            "rec_polys": [
                [[100, 9], [170, 9], [170, 21], [100, 21]],
                [[10, 12], [80, 12], [80, 24], [10, 24]],
                [[10, 40], [90, 40], [90, 52], [10, 52]],
            ],
        }]})()
        self.assertEqual(engine.ocr_image("unused.png"), "bên trái bên phải\ndòng sau")


if __name__ == "__main__":
    unittest.main()
