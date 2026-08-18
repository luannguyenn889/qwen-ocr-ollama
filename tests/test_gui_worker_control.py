import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.core import batch_ocr
from app.gui import run_gui
from app.gui.run_gui import AppGUI, OCRWorker, format_elapsed


class GuiWorkerControlTests(unittest.TestCase):
    def test_elapsed_time_log_format(self):
        self.assertEqual(format_elapsed(12.4), "12 giây")
        self.assertEqual(format_elapsed(125), "2 phút 5 giây")
        self.assertEqual(format_elapsed(3723), "1 giờ 2 phút 3 giây")

    def test_gui_uses_core_image_extraction_and_cleaning(self):
        self.assertIs(run_gui.extract_images_from_page, batch_ocr.extract_images_from_page)
        self.assertIs(run_gui.clean_markdown, batch_ocr.clean_markdown)
        self.assertIs(run_gui.ocr_qwen_images, batch_ocr.ocr_qwen_images)
        self.assertIs(run_gui.finalize_markdown, batch_ocr.finalize_markdown)
        self.assertEqual(run_gui.PROMPT, batch_ocr.PROMPT)
        self.assertEqual(run_gui.MODEL, batch_ocr.MODEL)
        sample = "```markdown\n| A | B |\n|---|---|\n| 1 | 2 |\n```"
        self.assertEqual(
            run_gui.finalize_markdown(run_gui.clean_markdown(sample)),
            batch_ocr.finalize_markdown(batch_ocr.clean_markdown(sample)),
        )

    def test_qwen_worker_count_is_clamped_to_two(self):
        worker = OCRWorker(
            Path("input.pdf"), Path("output"), queue.Queue(),
            threading.Event(), threading.Event(), "qwen3.5:4b", workers=4,
        )
        self.assertEqual(worker.workers, 2)

    def test_shared_qwen_request_guard_honors_cancellation(self):
        stop_event = threading.Event()
        resume_event = threading.Event()
        resume_event.set()
        worker = OCRWorker(
            Path("input.pdf"), Path("output"), queue.Queue(),
            stop_event, resume_event, "qwen3.5:4b",
        )

        worker._before_qwen_request()
        stop_event.set()
        with self.assertRaises(batch_ocr.PipelineCancelled):
            worker._before_qwen_request()

    def test_hybrid_label_resolves_to_qwen_model(self):
        self.assertEqual(
            batch_ocr.resolve_qwen_model("Hybrid (Paddle layout + qwen3.5:4b)"),
            batch_ocr.MODEL,
        )

    def test_stop_releases_paused_workers(self):
        app = AppGUI.__new__(AppGUI)
        app.stop_event = threading.Event()
        app.resume_event = threading.Event()
        app.log_text = Mock()
        app.btn_stop = Mock()

        app.stop_ocr()

        self.assertTrue(app.stop_event.is_set())
        self.assertTrue(app.resume_event.is_set())


if __name__ == "__main__":
    unittest.main()
