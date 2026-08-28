import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.core import batch_ocr
from app.gui import run_gui
from app.gui.run_gui import AppGUI, OCRWorker, completed_page_percent, format_elapsed


class GuiWorkerControlTests(unittest.TestCase):
    def test_elapsed_time_log_format(self):
        self.assertEqual(format_elapsed(12.4), "12 giây")
        self.assertEqual(format_elapsed(125), "2 phút 5 giây")
        self.assertEqual(format_elapsed(3723), "1 giờ 2 phút 3 giây")

    def test_page_ocr_progress_reserves_final_steps(self):
        self.assertEqual(completed_page_percent(0, 3), 0)
        self.assertEqual(completed_page_percent(3, 3), 90)

    def test_ollama_request_timeout_is_five_minutes(self):
        self.assertEqual(batch_ocr.OLLAMA_REQUEST_TIMEOUT_SECONDS, 300.0)

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
        self.assertTrue(worker.skip_blank_pages)
        self.assertEqual(worker.blank_detection_sensitivity, "safe")

    def test_blank_page_skipping_can_be_disabled(self):
        worker = OCRWorker(
            Path("input.pdf"), Path("output"), queue.Queue(),
            threading.Event(), threading.Event(), "qwen3.5:4b",
            skip_blank_pages=False,
        )
        self.assertFalse(worker.skip_blank_pages)

    def test_blank_page_sensitivity_accepts_vietnamese_gui_labels(self):
        worker = OCRWorker(
            Path("input.pdf"), Path("output"), queue.Queue(),
            threading.Event(), threading.Event(), "qwen3.5:4b",
            blank_detection_sensitivity="Mạnh mẽ",
        )
        self.assertEqual(worker.blank_detection_sensitivity, "aggressive")

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

    def test_ocr_worker_delegates_to_process_single_pdf(self):
        from unittest.mock import patch
        progress_q = queue.Queue()
        stop_event = threading.Event()
        resume_event = threading.Event()
        resume_event.set()

        fake_pdf = Path("test_doc.pdf")
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "is_dir", return_value=False), \
             patch("app.gui.run_gui.process_single_pdf") as mock_process:
            mock_process.return_value = Path("OCR/test_doc.md")

            worker = OCRWorker(
                fake_pdf, Path("OCR"), progress_q,
                stop_event, resume_event, "qwen3.5:4b", workers=1,
            )
            worker.run()

            self.assertTrue(mock_process.called)
            call_kwargs = mock_process.call_args[1]
            self.assertEqual(call_kwargs["pdf_path"], fake_pdf)
            self.assertEqual(call_kwargs["output_dir"], Path("OCR"))
            self.assertEqual(call_kwargs["stop_event"], stop_event)
            self.assertEqual(call_kwargs["resume_event"], resume_event)

        # Check queue events emitted
        events = []
        while not progress_q.empty():
            events.append(progress_q.get_nowait())

        event_types = [e[0] for e in events]
        self.assertIn("file_progress", event_types)
        self.assertIn("finished", event_types)
        self.assertEqual(events[-1], ("finished", "completed"))

    def test_app_gui_poll_queue_page_sub_progress(self):
        app = AppGUI.__new__(AppGUI)
        app.progress_queue = queue.Queue()
        app.page_progressbar = {}
        app.file_progressbar = {}
        app.page_progress_var = Mock()
        app.file_progress_var = Mock()
        app.stats_active_var = Mock()
        app._page_sub_progress = {}
        app._page_sub_labels = {}
        app._pages_done = 0
        app._total_pages = 2
        app._current_stage_text = "Chuẩn bị"
        app.log_text = Mock()
        app.root = Mock()
        app._is_running = True
        app.active_page_timers = {}

        # Put sub progress message
        app.progress_queue.put(("page_sub_progress", (1, 0.50, "Qwen Vision OCR")))
        app.poll_queue()

        self.assertAlmostEqual(app.page_progressbar["value"], 25.0)
        app.page_progress_var.set.assert_called_with("Trang: 25% (0/2 trang) — Qwen Vision OCR")


if __name__ == "__main__":
    unittest.main()


