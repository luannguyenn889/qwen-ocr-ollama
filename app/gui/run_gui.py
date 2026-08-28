"""
Module: run_gui.py
Nhiệm vụ: Giao diện đồ họa (GUI) quản lý tiến trình OCR tài liệu PDF sử dụng Tkinter.
Quy trình:
  1. Cho phép người dùng chọn tệp tin PDF hoặc thư mục chứa các tệp PDF.
  2. Bắt đầu/Dừng (Start/Stop) tiến trình OCR thông qua luồng chạy ngầm (Threading).
  3. Hiển thị tiến trình chi tiết của file, số trang hiện tại kèm thanh tiến trình ASCII và nhật ký (Log) thời gian thực.
"""

import os
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import threading
import queue
import tempfile
import time
import re
import warnings
import gc
from io import BytesIO
from pathlib import Path

# Third-party PaddleX/Protobuf code still uses deprecated datetime APIs on
# Windows. Keep those dependency warnings out of the end-user activity log.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\.protobuf(?:\..*)?")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"paddle(?:x|ocr)?(?:\..*)?")
warnings.filterwarnings(
    "ignore",
    message=r"datetime\.datetime\.utcfromtimestamp\(\) is deprecated.*",
    category=DeprecationWarning,
)

# pyrefly: ignore [missing-import]
from PIL import Image, ImageTk
# pyrefly: ignore [missing-import]
import pymupdf  # PyMuPDF
# pyrefly: ignore [missing-import]
from ollama import Client
from app.core.batch_ocr import (
    ENABLE_LAYOUT_DETECTION, MODEL, OLLAMA_REQUEST_TIMEOUT_SECONDS, PROMPT as CORE_PROMPT,
    TABLE_HTML_RETRY_INSTRUCTION, TABLE_RENDER_DPI, TABLE_SAFE_INSTRUCTION,
    TABLE_STRUCTURE_REPAIR_PROMPT, clean_markdown as core_clean_markdown,
    TEXT_ONLY_OUTPUT, warmup_qwen_model,
    QUALITY_RETRY_INSTRUCTION, apply_page_assets, classify_graphic_crop,
    cleanup_unreferenced_assets, deduplicate_exact_assets, quality_retry_instruction,
    extract_images_from_page as core_extract_images_from_page,
    finalize_markdown, format_finalization_report, link_extracted_images,
    merge_markdown_tables as core_merge_markdown_tables,
    needs_table_retry, normalize_worker_count, render_table_page, resolve_qwen_model,
    normalized_document_stem, ocr_coordinate_blocks, ocr_qwen_images,
    output_markdown_path, BlankOCRResult, PipelineCancelled, retain_extracted_image_blocks,
    classify_page_image, is_blank_pdf_page, is_blank_page_after_masking,
    is_blank_ocr_response, is_confirmed_signature_stamp,
    normalize_blank_detection_sensitivity, process_single_pdf,
)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

PROMPT = CORE_PROMPT

# The desktop frontend delegates OCR helpers and cleaning to the core pipeline.
merge_markdown_tables = core_merge_markdown_tables
extract_images_from_page = core_extract_images_from_page
clean_markdown = core_clean_markdown



def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for compact, readable GUI logs."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} giờ {minutes} phút {secs} giây"
    if minutes:
        return f"{minutes} phút {secs} giây"
    return f"{secs} giây"


def completed_page_percent(completed: int, total: int) -> float:
    """Reserve the final 10% for image review and atomic output writing."""
    if total <= 0:
        return 0.0
    return min(90.0, max(0.0, completed / total * 90.0))


class OCRWorker:
    def __init__(
        self, target_path: Path, output_dir: Path, progress_queue: queue.Queue,
        stop_event: threading.Event, resume_event: threading.Event,
        model_name: str, workers: int = 1, skip_blank_pages: bool = True,
        blank_detection_sensitivity: str = "safe",
        spell_correct: bool = False,
    ):
        self.workers = normalize_worker_count(workers, model_name)
        self.layout_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.pages_processed = 0
        self.target_path = target_path
        self.output_dir = output_dir
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        self.resume_event = resume_event
        self.model_name = model_name
        self.skip_blank_pages = bool(skip_blank_pages)
        self.blank_detection_sensitivity = normalize_blank_detection_sensitivity(
            blank_detection_sensitivity
        )
        self.spell_correct = bool(spell_correct)
        self.client = Client(
            host="http://localhost:11434", timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )

    def _before_qwen_request(self):
        """Wait while paused and abort cleanly when the user cancels."""
        self.resume_event.wait()
        if self.stop_event.is_set():
            raise PipelineCancelled()

    def run(self):
        batch_start_time = time.perf_counter()
        try:
            # 1. Thu thập danh sách tệp PDF
            pdf_files = []
            if self.target_path.is_file() and self.target_path.suffix.lower() == ".pdf":
                pdf_files.append(self.target_path)
            elif self.target_path.is_dir():
                pdf_files = sorted(list(self.target_path.glob("*.pdf")))

            if not pdf_files:
                self.progress_queue.put(("log", "Lỗi: Không tìm thấy file PDF nào để xử lý.\n"))
                self.progress_queue.put(("finished", "no_files"))
                return

            total_files = len(pdf_files)
            self.progress_queue.put(("log", f"Bắt đầu xử lý {total_files} file PDF...\n"))

            for file_idx, pdf_path in enumerate(pdf_files, 1):
                if self.stop_event.is_set():
                    break

                self.progress_queue.put(("file_progress", (file_idx - 1, total_files, pdf_path.name)))
                self.progress_queue.put(("log", f"\n[File {file_idx}/{total_files}] Đang xử lý: {pdf_path.name}\n"))
                pdf_start_time = time.perf_counter()

                try:
                    process_single_pdf(
                        pdf_path=pdf_path,
                        output_dir=self.output_dir,
                        client=self.client,
                        model_name=self.model_name,
                        workers=self.workers,
                        skip_blank_pages=self.skip_blank_pages,
                        blank_detection_sensitivity=self.blank_detection_sensitivity,
                        stop_event=self.stop_event,
                        resume_event=self.resume_event,
                        progress_callback=lambda event, data: self.progress_queue.put((event, data)),
                        spell_correct=self.spell_correct,
                    )
                except PipelineCancelled:
                    break

                self.progress_queue.put(("file_progress", (file_idx, total_files, pdf_path.name)))
                file_elapsed = time.perf_counter() - pdf_start_time
                self.progress_queue.put(("log", f"  - Tổng thời gian OCR file: {format_elapsed(file_elapsed)}\n"))
                gc.collect()

            if self.stop_event.is_set():
                batch_elapsed = time.perf_counter() - batch_start_time
                self.progress_queue.put(("log", f"\n[DỪNG] Tiến trình đã bị hủy bởi người dùng sau {format_elapsed(batch_elapsed)}.\n"))
                self.progress_queue.put(("finished", "cancelled"))
            else:
                batch_elapsed = time.perf_counter() - batch_start_time
                self.progress_queue.put(("log", f"\n[HOÀN THÀNH] Tổng thời gian OCR: {format_elapsed(batch_elapsed)}.\n"))
                self.progress_queue.put(("finished", "completed"))

        except Exception as e:
            batch_elapsed = time.perf_counter() - batch_start_time
            self.progress_queue.put(("log", f"\n[LỖI] Có lỗi xảy ra sau {format_elapsed(batch_elapsed)}: {str(e)}\n"))
            self.progress_queue.put(("finished", f"error: {str(e)}"))

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen OCR Pipeline & Manager")
        self.root.geometry("1200x900")
        self.root.minsize(900, 700)

        self.progress_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.is_paused = False
        self._is_running = False
        self.active_page_timers = {}
        self._page_sub_progress = {}
        self._page_sub_labels = {}
        self._pages_done = 0
        self._total_pages = 0
        self._current_stage_text = "Sẵn sàng"
        self.worker_thread = None
        self._preview_images: list[ImageTk.PhotoImage] = []
        self.latest_overlay_path: Path | None = None
        self._inline_overlay_image: ImageTk.PhotoImage | None = None

        self.create_styles()
        self.build_ui()
        self.root.after(100, self.poll_queue)

    def create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Dark premium theme styling
        style.configure(".", background="#f5f6f8", foreground="#333333", font=("Segoe UI", 10))
        style.configure("TFrame", background="#f5f6f8")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 14), foreground="#1a1a1a", background="#f5f6f8")
        style.configure("Status.TLabel", font=("Segoe UI", 10), background="#f5f6f8")
        style.configure("Action.TButton", font=("Segoe UI Semibold", 10), padding=6)
        style.configure("Start.TButton", background="#2ea44f", foreground="white")
        style.map("Start.TButton", background=[("active", "#2c974b")])
        style.configure("Stop.TButton", background="#cf222e", foreground="white")
        style.map("Stop.TButton", background=[("active", "#b31b26")])

    def build_ui(self):
        # Main layout frame
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Label
        header_lbl = ttk.Label(main_frame, text="Qwen OCR Document Pipeline", style="Header.TLabel")
        header_lbl.pack(anchor=tk.W, pady=(0, 8))

        # 1. File Selection Frame
        selection_frame = ttk.LabelFrame(main_frame, text=" Chọn tài liệu đầu vào ", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 8))

        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(selection_frame, textvariable=self.path_var, font=("Segoe UI", 10))
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        btn_file = ttk.Button(selection_frame, text="Chọn PDF...", command=self.browse_file, style="Action.TButton")
        btn_file.pack(side=tk.LEFT, padx=2)

        btn_dir = ttk.Button(selection_frame, text="Chọn thư mục...", command=self.browse_directory, style="Action.TButton")
        btn_dir.pack(side=tk.LEFT, padx=2)

        # 2. Output Directory Frame
        out_frame = ttk.LabelFrame(main_frame, text=" Thư mục đầu ra ", padding=10)
        out_frame.pack(fill=tk.X, pady=(0, 8))

        self.out_var = tk.StringVar(value=str(Path("OCR").resolve()))
        out_entry = ttk.Entry(out_frame, textvariable=self.out_var, font=("Segoe UI", 10))
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        btn_out = ttk.Button(out_frame, text="Thay đổi...", command=self.browse_output, style="Action.TButton")
        btn_out.pack(side=tk.LEFT)

        # 3. Model Selection
        model_frame = ttk.LabelFrame(main_frame, text=" Mô hình Qwen Vision (Paddle tự phân tích layout) ", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 8))

        models_list = []
        try:
            response = Client(host="http://localhost:11434", timeout=10.0).list()
            available_models = getattr(response, "models", None)
            if available_models is None and isinstance(response, dict):
                available_models = response.get("models", [])
            for model in available_models or []:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model")
                else:
                    name = getattr(model, "model", None) or getattr(model, "name", None)
                if name and name not in models_list:
                    models_list.append(name)
        except Exception:
            pass

        default_models = [
            MODEL,
            "Hybrid (Paddle layout + qwen3.5:4b)",
            "qwen2.5-vl:7b",
            "qwen2.5-vl:3b",
            "qwen2.5vl:7b",
        ]
        for model in default_models:
            if model not in models_list:
                models_list.append(model)

        self.model_var = tk.StringVar(value=MODEL)
        self.model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=models_list,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Label(model_frame, text="Luồng:").pack(side=tk.LEFT, padx=(5, 2))
        self.workers_var = tk.IntVar(value=1)
        workers_spin = ttk.Spinbox(model_frame, from_=1, to=2, textvariable=self.workers_var, width=5, font=("Segoe UI", 10))
        workers_spin.pack(side=tk.LEFT)

        self.skip_blank_pages_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            model_frame, text="Tự động bỏ qua trang trắng",
            variable=self.skip_blank_pages_var,
        ).pack(side=tk.LEFT, padx=(14, 0))

        ttk.Label(model_frame, text="Độ nhạy:").pack(side=tk.LEFT, padx=(10, 3))
        self.blank_sensitivity_var = tk.StringVar(value="An toàn")
        ttk.Combobox(
            model_frame,
            textvariable=self.blank_sensitivity_var,
            values=("An toàn", "Chuẩn", "Mạnh mẽ"),
            state="readonly",
            width=10,
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)

        self.spell_correct_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            model_frame, text="Sửa chính tả",
            variable=self.spell_correct_var,
        ).pack(side=tk.LEFT, padx=(10, 0))


        # 4. Status & Progress Indicators
        progress_frame = ttk.LabelFrame(main_frame, text=" Tiến trình ", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 8))

        self.file_progress_var = tk.StringVar(value="File: Sẵn sàng")
        self.file_lbl = ttk.Label(progress_frame, textvariable=self.file_progress_var, style="Status.TLabel")
        self.file_lbl.pack(anchor=tk.W, pady=2)

        self.file_progressbar = ttk.Progressbar(progress_frame, mode="determinate")
        self.file_progressbar.pack(fill=tk.X, pady=(2, 8))

        self.page_progress_var = tk.StringVar(value="Trang: Sẵn sàng")
        self.page_lbl = ttk.Label(progress_frame, textvariable=self.page_progress_var, style="Status.TLabel")
        self.page_lbl.pack(anchor=tk.W, pady=2)

        self.page_progressbar = ttk.Progressbar(progress_frame, mode="determinate")
        self.page_progressbar.pack(fill=tk.X, pady=(2, 5))

        # Thống kê tổng hợp (Summary Dashboard)
        stats_frame = ttk.LabelFrame(main_frame, text=" Thống kê tổng hợp ", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 8))

        self.stats_avg_var = tk.StringVar(value="Trung bình: -- s/trang")
        ttk.Label(stats_frame, textvariable=self.stats_avg_var).pack(side=tk.LEFT, expand=True)

        self.stats_eta_var = tk.StringVar(value="ETA: Đang tính toán...")
        ttk.Label(stats_frame, textvariable=self.stats_eta_var).pack(side=tk.LEFT, expand=True)

        self.stats_active_var = tk.StringVar(value="Đang OCR: --")
        ttk.Label(stats_frame, textvariable=self.stats_active_var).pack(side=tk.LEFT, expand=True)

        self.stats_warn_var = tk.StringVar(value="Cảnh báo: 0")
        ttk.Label(stats_frame, textvariable=self.stats_warn_var, foreground="red").pack(side=tk.LEFT, expand=True)

        # 5. Action Buttons (Start/Stop)
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(0, 8))
        action_frame.columnconfigure(0, weight=2)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)

        self.btn_start = ttk.Button(action_frame, text="Bắt đầu OCR", style="Start.TButton", command=self.start_ocr)
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.btn_pause = ttk.Button(action_frame, text="Tạm dừng", style="Action.TButton", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        self.btn_stop = ttk.Button(action_frame, text="Dừng lại", style="Stop.TButton", command=self.stop_ocr, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=2, sticky="ew")

        self.btn_preview = ttk.Button(
            action_frame, text="Xem trước Markdown...", style="Action.TButton", command=self.choose_preview
        )
        self.btn_preview.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(6, 0))
        self.btn_layout_preview = ttk.Button(
            action_frame, text="Xem overlay layout", style="Action.TButton",
            command=self.show_latest_layout, state=tk.DISABLED,
        )
        self.btn_layout_preview.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))


        # Live diagnostic preview: placed below controls so Start/Stop remain
        # visible even on smaller displays.
        diagnostics_frame = ttk.Frame(main_frame)
        diagnostics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        diagnostics_frame.columnconfigure(0, weight=3)
        diagnostics_frame.columnconfigure(1, weight=2)
        diagnostics_frame.rowconfigure(0, weight=1)

        overlay_frame = ttk.LabelFrame(diagnostics_frame, text=" Kiểm tra layout ", padding=8)
        overlay_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        overlay_header = ttk.Frame(overlay_frame)
        overlay_header.pack(fill=tk.X, pady=(0, 6))
        self.overlay_status_var = tk.StringVar(value="Chưa có trang nào được phân tích")
        ttk.Label(overlay_header, textvariable=self.overlay_status_var, style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(overlay_header, text="Bản xem nhanh; mở ảnh lớn để soi chi tiết", style="Status.TLabel").pack(side=tk.RIGHT)
        self.layout_canvas = tk.Canvas(overlay_frame, height=270, background="#252a34", highlightthickness=0)
        self.layout_canvas.pack(fill=tk.BOTH, expand=True)
        self.layout_canvas.bind("<Configure>", self._refresh_inline_layout_overlay)
        legend = ttk.Frame(overlay_frame)
        legend.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(legend, text="■ Text", foreground="#1688ff").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(legend, text="■ Ranh giới cột", foreground="#d98200").pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(legend, text="■ Bảng", foreground="#d62728").pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(diagnostics_frame, text=" Nhật ký hoạt động ", padding=6)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.log_text = ScrolledText(
            log_frame, height=16, font=("Consolas", 9), background="#1e1e1e",
            foreground="#d4d4d4", insertbackground="white", wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Chào mừng đến với Qwen OCR Manager. Vui lòng chọn file PDF để bắt đầu.\n")

        # Khởi động trước (Warm-up) model trên VRAM ở luồng ngầm
        def _startup_warmup():
            try:
                m_name = resolve_qwen_model(self.model_var.get().strip() or MODEL)
                client = Client(host="http://localhost:11434", timeout=15.0)
                warmup_qwen_model(client, m_name, keep_alive="30m")
            except Exception:
                pass
        threading.Thread(target=_startup_warmup, daemon=True).start()

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if filename:
            self.path_var.set(filename)

    def browse_directory(self):
        dir_name = filedialog.askdirectory()
        if dir_name:
            self.path_var.set(dir_name)

    def browse_output(self):
        dir_name = filedialog.askdirectory()
        if dir_name:
            self.out_var.set(dir_name)

    def choose_preview(self):
        filename = filedialog.askopenfilename(
            initialdir=self.out_var.get() or str(Path("OCR").resolve()),
            filetypes=[("Markdown files", "*.md")],
        )
        if filename:
            self.show_markdown_preview(Path(filename))

    def show_markdown_preview(self, markdown_path: Path):
        """Lightweight local Markdown preview with embedded extracted images."""
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Lỗi", f"Không thể mở Markdown:\n{error}")
            return

        window = tk.Toplevel(self.root)
        window.title(f"Xem trước — {markdown_path.name}")
        window.geometry("1000x760")
        preview = ScrolledText(window, wrap=tk.WORD, font=("Segoe UI", 12), background="white", foreground="#171717")
        preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        preview.tag_configure("h1", font=("Segoe UI Semibold", 20))
        preview.tag_configure("h2", font=("Segoe UI Semibold", 16))
        preview.tag_configure("h3", font=("Segoe UI Semibold", 14))
        preview.tag_configure("code", font=("Consolas", 10))

        # Keep PhotoImage objects alive for the lifetime of this window.
        images: list[ImageTk.PhotoImage] = []
        image_pattern = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
        lines = markdown.splitlines()
        table_widgets: list[ttk.Frame] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if (
                line.strip().startswith("|") and index + 1 < len(lines)
                and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", lines[index + 1])
            ):
                table_lines = [line]
                index += 2  # Skip the Markdown separator row.
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index])
                    index += 1
                table = self._insert_preview_table(preview, table_lines)
                table_widgets.append(table)
                preview.insert(tk.END, "\n")
                continue
            image_match = image_pattern.fullmatch(line.strip())
            if image_match:
                alt_text, relative_path = image_match.groups()
                image_path = markdown_path.parent / relative_path
                if image_path.is_file():
                    try:
                        with Image.open(image_path) as source:
                            image = source.convert("RGB")
                            image.thumbnail((900, 520))
                            photo = ImageTk.PhotoImage(image)
                        images.append(photo)
                        preview.image_create(tk.END, image=photo)
                        preview.insert(tk.END, "\n")
                        index += 1
                        continue
                    except OSError:
                        pass
                preview.insert(tk.END, f"[Không tải được ảnh: {relative_path}]\n", "code")
                index += 1
                continue

            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                preview.insert(tk.END, heading.group(2) + "\n", f"h{len(heading.group(1))}")
            elif not line.lstrip().startswith("<!--"):
                preview.insert(tk.END, line + "\n")
            index += 1
        preview.configure(state=tk.DISABLED)
        window._preview_images = images
        window._preview_tables = table_widgets

    @staticmethod
    def _insert_preview_table(preview: ScrolledText, markdown_rows: list[str]) -> ttk.Frame:
        """Embed a real Tk table for a Markdown table block in the preview."""
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in markdown_rows]
        column_count = max(len(row) for row in rows)
        for row in rows:
            row.extend([""] * (column_count - len(row)))
        frame = ttk.Frame(preview, padding=(2, 4))
        columns = [f"c{index}" for index in range(column_count)]
        table = ttk.Treeview(frame, columns=columns, show="headings", height=min(max(len(rows), 2), 10))
        for index, column in enumerate(columns):
            values = [row[index] for row in rows]
            width = min(max(max((len(value) for value in values), default=8) * 8, 90), 300)
            table.heading(column, text=rows[0][index])
            table.column(column, width=width, minwidth=70, stretch=True, anchor=tk.W)
        for row in rows[1:]:
            table.insert("", tk.END, values=row)
        table.pack(fill=tk.X, expand=True)
        preview.window_create(tk.END, window=frame, stretch=True)
        return frame

    def show_latest_layout(self):
        if not self.latest_overlay_path or not self.latest_overlay_path.is_file():
            messagebox.showinfo("Overlay layout", "Chưa có overlay layout để hiển thị.")
            return
        window = tk.Toplevel(self.root)
        window.title(f"Overlay layout — {self.latest_overlay_path.name}")
        window.geometry("1050x780")
        canvas = tk.Canvas(window, background="#2b2b2b")
        canvas.pack(fill=tk.BOTH, expand=True)
        try:
            with open(self.latest_overlay_path, "rb") as f:
                img_bytes = f.read()
            with Image.open(BytesIO(img_bytes)) as source:
                image = source.convert("RGB")
                image.thumbnail((1000, 720))
                photo = ImageTk.PhotoImage(image)
            canvas.create_image(10, 10, anchor=tk.NW, image=photo)
            canvas.create_text(10, image.height + 25, anchor=tk.W, fill="white",
                               text="Xanh dương: vùng text | Cam: ranh giới cột | Đỏ: vùng bảng")
            window._overlay_image = photo
        except OSError as error:
            messagebox.showerror("Lỗi", f"Không thể tải overlay:\n{error}")

    def update_inline_layout_overlay(self, overlay_path: Path):
        """Render the latest diagnostic image inside the main GUI."""
        try:
            with open(overlay_path, "rb") as f:
                img_bytes = f.read()
            with Image.open(BytesIO(img_bytes)) as source:
                image = source.convert("RGB")
                max_width = max(self.layout_canvas.winfo_width() - 24, 300)
                max_height = max(self.layout_canvas.winfo_height() - 14, 180)
                image.thumbnail((max_width, max_height))
                photo = ImageTk.PhotoImage(image)
        except OSError as error:
            self.log_text.insert(tk.END, f"[Chú ý] Không hiển thị được overlay: {error}\n")
            return
        self.layout_canvas.delete("all")
        canvas_width = max(self.layout_canvas.winfo_width(), image.width + 20)
        self.layout_canvas.create_image(canvas_width // 2, 8, anchor=tk.N, image=photo)
        self._inline_overlay_image = photo
        self.overlay_status_var.set(f"Đang xem: {overlay_path.name}")

    def _refresh_inline_layout_overlay(self, _event=None):
        if self.latest_overlay_path and self.latest_overlay_path.is_file():
            self.root.after_idle(lambda: self.update_inline_layout_overlay(self.latest_overlay_path))
        else:
            self._draw_empty_layout_state()

    def _draw_empty_layout_state(self):
        if not hasattr(self, "layout_canvas"):
            return
        canvas = self.layout_canvas
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        canvas.delete("all")
        canvas.create_text(width // 2, height // 2 - 12, anchor=tk.CENTER, fill="#edf1f7",
                           font=("Segoe UI Semibold", 13), text="Chưa có layout để xem")
        canvas.create_text(width // 2, height // 2 + 16, anchor=tk.CENTER, fill="#aeb9c9",
                           font=("Segoe UI", 10), text="Chọn PDF và bấm “Bắt đầu OCR”; overlay sẽ xuất hiện theo từng trang.")

    def start_ocr(self):
        target = self.path_var.get().strip()
        output = self.out_var.get().strip()

        if not target:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn tệp PDF hoặc thư mục chứa PDF.")
            return

        target_path = Path(target)
        if not target_path.exists():
            messagebox.showerror("Lỗi", "Đường dẫn đầu vào không tồn tại.")
            return

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_pause.configure(state=tk.NORMAL, text="Tạm dừng")
        self._is_running = True
        self.is_paused = False
        self.resume_event.set()
        self.active_page_timers = {}
        self._page_sub_progress = {}
        self._page_sub_labels = {}
        self._pages_done = 0
        self._total_pages = 0
        self._current_stage_text = "Đang chuẩn bị"
        self.file_progressbar["value"] = 0
        self.page_progressbar["value"] = 0
        self.file_progress_var.set("Tập tin: Đang khởi tạo...")
        self.page_progress_var.set("Trang: Đang chuẩn bị (0%)...")
        self.stats_active_var.set("Đang OCR: đang chuẩn bị...")

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "Đang khởi tạo...\n")
        self.log_text.see(tk.END)

        self.stop_event.clear()

        model_name = self.model_var.get().strip() or MODEL
        self.log_text.insert(tk.END, f"Mô hình được chọn: {model_name}\n")
        resolved_model = resolve_qwen_model(model_name)
        if resolved_model != model_name:
            self.log_text.insert(tk.END, f"Dùng Ollama model: {resolved_model}\n")
        worker = OCRWorker(
            target_path, Path(output), self.progress_queue, self.stop_event,
            self.resume_event, model_name,
            workers=self.workers_var.get() if hasattr(self, 'workers_var') else 1,
            skip_blank_pages=(
                self.skip_blank_pages_var.get()
                if hasattr(self, "skip_blank_pages_var") else True
            ),
            blank_detection_sensitivity=(
                self.blank_sensitivity_var.get()
                if hasattr(self, "blank_sensitivity_var") else "safe"
            ),
            spell_correct=(
                self.spell_correct_var.get()
                if hasattr(self, "spell_correct_var") else False
            ),
        )
        self.worker_thread = threading.Thread(target=worker.run, daemon=True)
        self.worker_thread.start()


    def toggle_pause(self):
        if not self._is_running: return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.resume_event.clear()
            self.btn_pause.configure(text="Tiếp tục")
            self.progress_queue.put(("stage_status", "Đang tạm dừng"))
            self.progress_queue.put(("log", "\n[TẠM DỪNG] Tiến trình đang tạm dừng...\n"))
        else:
            self.resume_event.set()
            self.btn_pause.configure(text="Tạm dừng")
            self.progress_queue.put(("stage_status", "Đang tiếp tục xử lý"))
            self.progress_queue.put(("log", "\n[TIẾP TỤC] Tiến trình tiếp tục chạy...\n"))

    def stop_ocr(self):
        self.stop_event.set()
        # Release workers blocked at resume_event.wait() so they can observe the
        # stop flag and exit even when Stop is pressed while paused.
        self.resume_event.set()
        self.log_text.insert(tk.END, "\nĐang yêu cầu dừng tiến trình... Vui lòng đợi.\n")
        self.log_text.see(tk.END)
        self.btn_stop.configure(state=tk.DISABLED)

    def make_ascii_bar(self, current, total, width=20) -> str:
        if total <= 0:
            return "░" * width
        filled_len = int(round(width * current / float(total)))
        bar = "█" * filled_len + "░" * (width - filled_len)
        return bar

    def poll_queue(self):
        while not self.progress_queue.empty():
            try:
                msg_type, data = self.progress_queue.get_nowait()
                if msg_type == "log":
                    if any(marker in data for marker in ("[Chú ý]", "[Cảnh báo]", "[Warning]", "[LỖI]")):
                        self.warning_count = getattr(self, "warning_count", 0) + 1
                        if hasattr(self, "stats_warn_var"):
                            self.stats_warn_var.set(f"Cảnh báo: {self.warning_count}")
                    self.log_text.insert(tk.END, data)
                    self.log_text.see(tk.END)
                elif msg_type == "stats_update":
                    avg_time, eta_seconds = data
                    if hasattr(self, "stats_avg_var"):
                        self.stats_avg_var.set(f"Trung bình: {avg_time:.1f} s/trang")
                        if eta_seconds > 0:
                            m, s = divmod(int(eta_seconds), 60)
                            self.stats_eta_var.set(f"ETA: ~{m} phút {s} giây")
                        else:
                            self.stats_eta_var.set("ETA: Sắp xong")
                elif msg_type == "file_progress":
                    curr, total, name = data
                    percent = (curr / total) * 100 if total > 0 else 0
                    self.file_progress_var.set(f"Tập tin {curr}/{total}: {name} ({int(percent)}%)")
                    self.file_progressbar["value"] = percent
                    self._page_sub_progress = {}
                    self._page_sub_labels = {}
                    self._pages_done = 0
                    self._total_pages = 0
                elif msg_type == "page_sub_progress":
                    page_num, sub_ratio, stage_desc = data
                    self._page_sub_progress[page_num] = max(
                        self._page_sub_progress.get(page_num, 0.0), float(sub_ratio)
                    )
                    self._page_sub_labels[page_num] = stage_desc
                    total_p = getattr(self, "_total_pages", 0)
                    done_p = getattr(self, "_pages_done", 0)
                    if total_p > 0:
                        sub_sum = sum(self._page_sub_progress.values())
                        pct = min(93.0, (sub_sum / total_p) * 100.0)
                        self.page_progressbar["value"] = pct
                        self.page_progress_var.set(
                            f"Trang: {int(pct)}% ({done_p}/{total_p} trang) — {stage_desc}"
                        )
                elif msg_type == "page_progress":
                    curr, total = data
                    self._pages_done = curr
                    self._total_pages = total
                    if curr in self._page_sub_progress:
                        self._page_sub_progress[curr] = 1.0
                    percent = (curr / total) * 100 if total > 0 else 0
                    val = min(93.0, percent) if percent < 100 else 100.0
                    self.page_progressbar["value"] = val
                    self.page_progress_var.set(f"Trang: {int(percent)}% ({curr}/{total} trang hoàn tất)")
                elif msg_type == "stage_status":
                    self._current_stage_text = data
                    total_p = getattr(self, "_total_pages", 0)
                    done_p = getattr(self, "_pages_done", 0)
                    pct = self.page_progressbar["value"]
                    if total_p > 0:
                        self.page_progress_var.set(
                            f"Trang: {int(pct)}% ({done_p}/{total_p} trang) — {data}"
                        )
                    else:
                        self.page_progress_var.set(f"Giai đoạn: {data}")
                    if data == "Đang tạm dừng":
                        self.stats_active_var.set("Trạng thái: Đang tạm dừng")
                elif msg_type == "stage_progress":
                    pct = min(100.0, max(0.0, float(data)))
                    self.page_progressbar["value"] = pct
                    total_p = getattr(self, "_total_pages", 0)
                    done_p = getattr(self, "_pages_done", total_p)
                    stage = getattr(self, "_current_stage_text", "Đang xử lý")
                    if total_p > 0:
                        self.page_progress_var.set(
                            f"Trang: {int(pct)}% ({done_p}/{total_p} trang) — {stage}"
                        )
                    else:
                        self.page_progress_var.set(f"Tiến trình: {int(pct)}% — {stage}")
                elif msg_type == "page_timer_start":
                    page_num, started_at = data
                    if not hasattr(self, "active_page_timers"):
                        self.active_page_timers = {}
                    self.active_page_timers[page_num] = started_at
                elif msg_type == "page_timer_end":
                    if hasattr(self, "active_page_timers"):
                        self.active_page_timers.pop(data, None)
                elif msg_type == "layout_overlay":
                    self.latest_overlay_path = Path(data)
                    self.btn_layout_preview.configure(state=tk.NORMAL)
                    self.update_inline_layout_overlay(self.latest_overlay_path)
                elif msg_type == "finished":
                    self.btn_start.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    self.btn_pause.configure(state=tk.DISABLED)
                    self._is_running = False
                    self.active_page_timers = {}
                    self.stats_active_var.set("Đang OCR: --")
                    if data == "completed":
                        self.file_progressbar["value"] = 100
                        self.page_progressbar["value"] = 100
                    else:
                        self.file_progressbar["value"] = 0
                        self.page_progressbar["value"] = 0
                    self.file_progress_var.set("File: Xong")
                    self.page_progress_var.set("Trang: Xong")

                    if data == "completed":
                        messagebox.showinfo("Thông báo", "Quá trình OCR đã hoàn thành thành công!")
                    elif data == "cancelled":
                        messagebox.showwarning("Thông báo", "Quá trình OCR đã bị hủy.")
                    elif data.startswith("error"):
                        messagebox.showerror("Lỗi", f"Quá trình chạy gặp lỗi:\n{data}")
            except Exception:
                pass
        active_timers = getattr(self, "active_page_timers", {})
        if getattr(self, "is_paused", False) and self._is_running:
            self.stats_active_var.set("Trạng thái: Đang tạm dừng")
        elif active_timers and self._is_running:
            now = time.perf_counter()
            running = " | ".join(
                f"Trang {page_num}: {now - started_at:.1f}s"
                for page_num, started_at in sorted(active_timers.items())
            )
            self.stats_active_var.set(f"Đang OCR: {running}")
        elif hasattr(self, "stats_active_var") and self._is_running:
            self.stats_active_var.set("Đang OCR: đang chuẩn bị...")
        self.root.after(100, self.poll_queue)

def run_gui():
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()


def main():
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
