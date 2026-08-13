"""
Module: run_gui.py
Nhiệm vụ: Giao diện đồ họa (GUI) quản lý tiến trình OCR tài liệu PDF sử dụng Tkinter.
Quy trình: 
  1. Cho phép người dùng chọn tệp tin PDF hoặc thư mục chứa các tệp PDF.
  2. Bắt đầu/Dừng (Start/Stop) tiến trình OCR thông qua luồng chạy ngầm (Threading).
  3. Hiển thị tiến trình chi tiết của file, số trang hiện tại kèm thanh tiến trình ASCII và nhật ký (Log) thời gian thực.
"""

import os
import threading
import queue
import tempfile
import time
import re
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image, ImageTk 
# pyrefly: ignore [missing-import]
import pymupdf  # PyMuPDF 
# pyrefly: ignore [missing-import]
from ollama import Client
from app.core.batch_ocr import (
    ENABLE_LAYOUT_DETECTION, TABLE_HTML_RETRY_INSTRUCTION, TABLE_RENDER_DPI, TABLE_SAFE_INSTRUCTION,
    TABLE_STRUCTURE_REPAIR_PROMPT, clean_markdown as core_clean_markdown,
    QUALITY_RETRY_INSTRUCTION, apply_page_assets,
    extract_images_from_page as core_extract_images_from_page,
    finalize_markdown, link_extracted_images,
    needs_table_retry, normalize_worker_count, page_is_tiled_scan, render_table_page, resolve_qwen_model,
    ocr_coordinate_blocks, ocr_qwen_images, PipelineCancelled,
)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

MODEL = "qwen3.5:4b"
PROMPT = """
Convert this scanned document to Markdown format according to the following requirements:

1. Remove unnecessary elements:
   - Automatically detect and remove headers, footers, footnotes, and page numbers to make the content cleaner and clearer.

2. Preserve the original document structure and layout:
   - Maintain heading hierarchies (Heading #, ##, ###, ####), paragraphs, and lists (bulleted/numbered lists).
   - CRITICAL: All question headers (e.g., "Câu 1.", "Câu 2.", "Câu 10.") must be strictly formatted as Heading Level 4: `#### Câu X.`. Do not use bold tags (`**Câu X.**`) or normal text for question headers.
   - Ensure the correct natural reading order of the document. For multi-column layouts or multiple-choice questions side-by-side, read column-by-column and keep the options (A, B, C, D) associated with their correct question. Do not merge text across columns.

3. Recognize special components:
   - Mathematical expressions: Automatically convert all formulas, variables, subscripts (e.g., S_1 to $S_1$), primes (e.g., A' to $A'$), fractions (always use \\frac{num}{den} for vertical fractions), and math symbols into LaTeX format. Use $...$ for inline formulas and $$...$$ for block formulas. Do not repeat characters or duplicate terms.
      * CRITICAL: Never use HTML space entities (such as &nbsp; or &amp;nbsp;) anywhere in the document. Use standard markdown spaces or newlines to separate options. Each mathematical expression must have its own closed pair of dollar signs ($). Never group non-mathematical text, punctuation, labels (like 'B.', 'C.', 'D.') inside a dollar sign pair.
   - Tables: Extract tables accurately. For complex tables (with merged cells or multi-tiered headers), export them as HTML tables (<table>). For simple tables, use the standard Markdown table format.
   - Images and captions: If images or diagrams are detected, represent them as a Markdown image tag with the description inside the square brackets and a placeholder path inside the parentheses, for example: `![Description of the image/diagram](image_placeholder.png)`. Never put descriptive text inside the parentheses.

4. Vietnamese Language Corrections (Spelling & Diacritics):
   - Correct any spelling, typographic, or diacritic errors in Vietnamese text. Pay special attention to accents/diacritics to ensure the output is grammatically correct and meaningful in Vietnamese context (for example, correct "vấn kiện" / "vấn kiến" to "văn kiện", "chương nghị sự" to "chương trình nghị sự").

5. General requirements:
   - Do not summarize the content, keep both Vietnamese and English text exactly as written.
   - Return only the raw Markdown content, do not wrap it inside ```markdown code blocks.
""".strip()


def extract_images_from_page(pdf_path: Path, page_index: int, output_img_dir: Path, prefix: str) -> list[str]:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    image_list = page.get_images(full=True)
    tiled_scan = page_is_tiled_scan(pdf_path, page_index)
    if not tiled_scan:
        image_list = sorted(
            image_list,
            key=lambda info: min(
                ((rect.y0, rect.x0) for rect in page.get_image_rects(info[0])),
                default=(float("inf"), float("inf")),
            ),
        )
    
    extracted_paths = []
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Trích xuất ảnh raster nhúng sẵn
    for img_idx, img_info in enumerate(() if tiled_scan else image_list, 1):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            img_name = f"{prefix}_page_{page_index + 1}_img_{img_idx}.{image_ext}"
            img_path = output_img_dir / img_name
            img_path.write_bytes(image_bytes)
            
            extracted_paths.append(f"images/{img_name}")
        except Exception as e:
            print(f"Error extracting image xref {xref} on page {page_index}: {e}")
            
    # 2. Phát hiện và crop các cụm hình vẽ vector (diagrams/drawings)
    try:
        drawings = page.get_drawings()
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        candidate_rects = []
        for d in drawings:
            r = d["rect"]
            if r.is_empty:
                continue
            # Bỏ qua phần header (12% trên) và footer (12% dưới)
            if r.y1 < page_height * 0.12 or r.y0 > page_height * 0.88:
                continue
            # Bỏ qua các đường kẻ viền chiếm gần hết chiều ngang hoặc dọc trang
            if r.width > page_width * 0.9 or r.height > page_height * 0.9:
                continue
            candidate_rects.append(r)
            
        if candidate_rects:
            # Gom nhóm các rect đè nhau hoặc ở gần nhau
            threshold = 30
            merged = []
            for r in candidate_rects:
                placed = False
                for idx, m in enumerate(merged):
                    dilated_m = pymupdf.Rect(m.x0 - threshold, m.y0 - threshold, m.x1 + threshold, m.y1 + threshold)
                    if dilated_m.intersects(r):
                        merged[idx] = m | r
                        placed = True
                        break
                if not placed:
                    merged.append(r)
            
            # Gom nhóm đệ quy cho đến khi không gộp thêm được nữa
            changed = True
            while changed:
                changed = False
                new_merged = []
                for r in merged:
                    placed = False
                    for idx, nm in enumerate(new_merged):
                        dilated_nm = pymupdf.Rect(nm.x0 - threshold, nm.y0 - threshold, nm.x1 + threshold, nm.y1 + threshold)
                        if dilated_nm.intersects(r):
                            new_merged[idx] = nm | r
                            placed = True
                            changed = True
                            break
                    if not placed:
                        new_merged.append(r)
                merged = new_merged
            
            # Ignore vector clusters that are actually glyph-based page text.
            text_word_centres = [
                pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
                for word in page.get_text("words")
            ]
            # Render và lưu từng cụm diagram
            for c_idx, rect in enumerate(merged, len(image_list) + 1):
                if sum(rect.contains(centre) for centre in text_word_centres) >= 30:
                    continue
                padding = 10
                crop_rect = pymupdf.Rect(
                    max(0, rect.x0 - padding),
                    max(0, rect.y0 - padding),
                    min(page_width, rect.x1 + padding),
                    min(page_height, rect.y1 + padding)
                )
                zoom = 300 / 72
                mat = pymupdf.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                
                img_name = f"{prefix}_page_{page_index + 1}_draw_{c_idx}.png"
                img_path = output_img_dir / img_name
                pix.save(str(img_path))
                
                extracted_paths.append(f"images/{img_name}")
    except Exception as dev_err:
        print(f"Error extracting vector drawings on page {page_index}: {dev_err}")
            
    doc.close()
    return extracted_paths

def merge_markdown_tables(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    blocks = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        is_table_start = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") > 1
        if is_table_start and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            is_sep = next_stripped.startswith("|") and next_stripped.endswith("|") and all(c in " -:+|" for c in next_stripped)
            if is_sep:
                table_rows = []
                table_comments = []
                while i < len(lines):
                    curr_line = lines[i]
                    curr_stripped = curr_line.strip()
                    if curr_stripped.startswith("<!--") and curr_stripped.endswith("-->"):
                        table_comments.append(curr_line)
                        i += 1
                        continue
                    if curr_stripped.startswith("|") and curr_stripped.endswith("|") and curr_stripped.count("|") > 1:
                        table_rows.append(curr_line)
                        i += 1
                    else:
                        break
                blocks.append({
                    "type": "table",
                    "rows": table_rows,
                    "comments": table_comments
                })
                continue
                
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            blocks.append({"type": "comment", "line": line})
        else:
            blocks.append({"type": "text", "line": line})
        i += 1

    merged_blocks = []
    for block in blocks:
        if block["type"] == "table":
            found_table_idx = None
            only_whitespace_or_comments = True
            for idx in range(len(merged_blocks) - 1, -1, -1):
                prev = merged_blocks[idx]
                if prev["type"] == "table":
                    found_table_idx = idx
                    break
                elif prev["type"] == "comment":
                    continue
                elif prev["type"] == "text" and prev["line"].strip() == "":
                    continue
                else:
                    only_whitespace_or_comments = False
                    break
            
            if found_table_idx is not None and only_whitespace_or_comments:
                last = merged_blocks[found_table_idx]
                header_last = [c.strip().lower() for c in last["rows"][0].split("|")[1:-1]]
                header_curr = [c.strip().lower() for c in block["rows"][0].split("|")[1:-1]]
                
                if header_last == header_curr:
                    data_rows = block["rows"][2:]
                    last["rows"].extend(data_rows)
                    last["comments"].extend(block["comments"])
                    # Remove the skipped whitespace/comment blocks in between
                    del merged_blocks[found_table_idx + 1:]
                    continue
                    
        merged_blocks.append(block)
        
    output = []
    for block in merged_blocks:
        if block["type"] == "text":
            output.append(block["line"])
        elif block["type"] == "comment":
            output.append(block["line"])
        elif block["type"] == "table":
            output.extend(block["rows"])
            output.extend(block["comments"])
            
    return "\n".join(output)

def clean_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Remove a model acknowledgement before the actual document Markdown.
    text = re.sub(
        r"\A(?:based on (?:your|the) requirements|here (?:is|are) (?:the )?(?:converted|extracted|requested)|dưới đây là).+?(?:\r?\n){2,}",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def unwrap_prose_math(match):
        content = match.group(1)
        words = content.split()
        math_operators = len(re.findall(r"[=+*/^_{}\\]", content))
        if len(words) >= 6 and math_operators <= 2:
            return content
        return match.group(0)

    return re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", unwrap_prose_math, text)

def post_process_markdown(text: str) -> str:
    # 1. Clean HTML entities like &nbsp; and duplicate spaces
    text = re.sub(r'&(?:nbsp|amp);', ' ', text)
    text = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: match.group(1)
        if len(match.group(1).split()) >= 6
        and len(re.findall(r"[=+*/^_{}\\]", match.group(1))) <= 2
        else match.group(0),
        text,
    )
    
    processed_lines = []
    for line in text.splitlines():
        # 2. Fix merged math blocks that span across option transitions (e.g. A. $expr1 B. expr2$)
        # This split logic checks if a math block contains option transitions and breaks it up generically.
        def split_merged_math(match):
            content = match.group(1)
            # Find occurrences of standard uppercase list/option labels followed by a dot or parenthesis,
            # e.g., " B. ", " C. ", " 2) ", " b. "
            opt_transition = re.search(r'\s+([A-Za-z0-9])[\.\)]\s+', content)
            if opt_transition:
                # Split at option labels to keep math expressions separated
                parts = re.split(r'(\s+[A-Za-z0-9][\.\)]\s+)', content)
                new_parts = []
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        part_stripped = part.strip()
                        if part_stripped:
                            new_parts.append(f"${part_stripped}$")
                    else:
                        new_parts.append(part)
                return "".join(new_parts)
            return match.group(0)
            
        line = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', split_merged_math, line)
        
        # 3. Balance unescaped dollar signs on each line
        # If there's an odd number of dollar signs, close the last one at the end of the line (before punctuation)
        dollar_indices = [i for i, char in enumerate(line) if char == '$' and (i == 0 or line[i-1] != '\\')]
        if len(dollar_indices) % 2 != 0:
            line_stripped = line.rstrip()
            if line_stripped.endswith('.'):
                line = line_stripped[:-1] + '$.'
            else:
                line = line_stripped + '$'
                
        processed_lines.append(line)
        
    result = "\n".join(processed_lines)
    return re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: match.group(1)
        if len(match.group(1).split()) >= 6
        and len(re.findall(r"[=+*/^_{}\\]", match.group(1))) <= 2
        else match.group(0),
        result,
    )


# The desktop frontend deliberately delegates OCR helpers to the core pipeline.
# Legacy local definitions above remain temporarily for source compatibility but
# are not used by OCRWorker.
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


class OCRWorker:
    def __init__(self, target_path: Path, output_dir: Path, progress_queue: queue.Queue, stop_event: threading.Event, resume_event: threading.Event, model_name: str, workers: int = 1):
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
        self.client = Client(host="http://localhost:11434")

    def run(self):
        batch_start_time = time.perf_counter()
        try:
            # 1. Collect PDFs
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
            from app.core.formula_ocr import formula_ocr_status
            _, formula_status = formula_ocr_status()
            self.progress_queue.put(("log", f"Trạng thái Formula OCR: {formula_status}.\n"))

            for file_idx, pdf_path in enumerate(pdf_files, 1):
                self.pages_processed = 0
                pdf_start_time = time.perf_counter()
                if self.stop_event.is_set():
                    break
                
                self.progress_queue.put(("file_progress", (file_idx, total_files, pdf_path.name)))
                self.progress_queue.put(("log", f"\n[File {file_idx}/{total_files}] Đang xử lý: {pdf_path.name}\n"))
                
                # Create output folder
                self.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = self.output_dir / f"{pdf_path.stem}.md"
                
                # Render pages
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    
                    self.progress_queue.put(("log", "  - Đang render PDF thành hình ảnh...\n"))
                    doc = pymupdf.open(pdf_path)
                    total_pages = len(doc)
                    
                    page_images = []
                    render_timings = []
                    for idx in range(total_pages):
                        if self.stop_event.is_set():
                            break
                        render_started_at = time.perf_counter()
                        page = doc.load_page(idx)
                        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
                        img_path = temp_dir_path / f"page_{idx + 1}.png"
                        pix.save(str(img_path))
                        page_images.append(img_path)
                        render_timings.append(time.perf_counter() - render_started_at)
                    doc.close()
                    
                    if self.stop_event.is_set():
                        break

                    # OCR each page
                    ocr_contents = []
                    is_hybrid = "hybrid" in self.model_name.lower()
                    hybrid_model = resolve_qwen_model(self.model_name)
                    layout_detector = None
                    if ENABLE_LAYOUT_DETECTION:
                        try:
                            from app.core.layout_detector import create_layout_detector
                            layout_detector, layout_status = create_layout_detector()
                            self.progress_queue.put(("log", f"  - Trạng thái layout: {layout_status}.\n"))
                        except Exception as layout_error:
                            self.progress_queue.put(("log", f"  - Trạng thái layout: disabled; dùng nguyên trang ({layout_error}).\n"))

                    def process_page_worker(args):
                        idx, img_path = args
                        if self.stop_event.is_set():
                            return None
                        
                        page_num = idx + 1
                        self.progress_queue.put(("page_progress", (page_num, total_pages)))
                        self.progress_queue.put(("log", f"  - Đang OCR Trang {page_num}/{total_pages}...\n"))
                        
                        started_at = time.perf_counter()
                        self.progress_queue.put(("page_timer_start", (page_num, started_at)))
                        paddle_seconds = 0.0
                        qwen_seconds = 0.0
                        extracted_paths = []
                        try:
                            output_img_dir = self.output_dir / "images"
                            from app.core.batch_ocr import extract_images_from_page
                            with self.layout_lock:
                                extracted_paths = extract_images_from_page(
                                    pdf_path, idx, output_img_dir, pdf_path.stem,
                                    layout_detector=layout_detector,
                                    page_image_path=img_path
                                )
                        except Exception as img_err:
                            self.progress_queue.put(("log", f"    -> [Chú ý] Không thể trích xuất ảnh: {img_err}\n"))

                        qwen_images = [img_path]
                        has_table = False
                        ordered_blocks = []
                        if layout_detector is not None:
                            try:
                                paddle_started_at = time.perf_counter()
                                with self.layout_lock:
                                    columns, has_table, text_regions, table_regions = layout_detector.analyse_with_regions(img_path)
                                    ordered_blocks = layout_detector.detect_ordered_blocks(img_path.read_bytes())
                                paddle_seconds = time.perf_counter() - paddle_started_at
                                from app.core.layout_detector import save_layout_overlay
                                overlay_path = save_layout_overlay(
                                    img_path,
                                    self.output_dir / "layout_debug" / f"{pdf_path.stem}_page_{page_num}_layout.png",
                                    text_regions, columns, table_regions,
                                )
                                self.progress_queue.put(("layout_overlay", overlay_path))
                                if len(columns) >= 2:
                                    from app.core.layout_detector import crop_segments
                                    qwen_images = crop_segments(img_path, columns, temp_dir_path / "columns")
                                    self.progress_queue.put(("log", f"    -> Phát hiện {len(qwen_images)} cột, đọc từ trái sang phải.\n"))
                                if has_table:
                                    qwen_images = [render_table_page(pdf_path, idx, temp_dir_path / "table_pages")]
                                    self.progress_queue.put(("log", f"    -> Phát hiện bảng, render lại {TABLE_RENDER_DPI} DPI trước khi gửi Qwen.\n"))
                            except Exception as layout_error:
                                self.progress_queue.put(("log", f"    -> [Chú ý] Layout lỗi, dùng nguyên trang: {layout_error}\n"))
                                qwen_images, has_table = [img_path], False
                                ordered_blocks = []

                        qwen_started_at = time.perf_counter()

                        def before_qwen_request():
                            self.resume_event.wait()
                            if self.stop_event.is_set():
                                raise PipelineCancelled()

                        try:
                            selected_model = hybrid_model if is_hybrid else resolve_qwen_model(self.model_name)
                            page_md = ocr_coordinate_blocks(
                                self.client, selected_model, img_path, ordered_blocks, extracted_paths,
                                temp_dir_path / "layout_blocks" / f"page_{page_num}",
                                log_func=lambda message: self.progress_queue.put(("log", f"    -> [Trang {page_num}] {message}\n")),
                                before_request=before_qwen_request,
                            )
                            if page_md is not None:
                                self.progress_queue.put(("log", "    -> Đã OCR toàn trang một lần và đặt ảnh theo reading order.\n"))
                            else:
                                page_md = ocr_qwen_images(
                                    self.client, selected_model, qwen_images, has_table=has_table,
                                    log_func=lambda message: self.progress_queue.put(("log", f"    -> [Trang {page_num}] {message}\n")),
                                    before_request=before_qwen_request,
                                )
                        except PipelineCancelled:
                            return None
                        qwen_seconds = time.perf_counter() - qwen_started_at
                        
                        if extracted_paths:
                            self.progress_queue.put(("log", f"    -> Đã trích xuất {len(extracted_paths)} hình ảnh từ PDF trang {page_num}.\n"))
                        page_md = apply_page_assets(page_md, page_num, extracted_paths)

                        from app.core.quality_gate import choose_best_page, evaluate_page
                        quality = evaluate_page(page_md, self.output_dir)
                        if quality.warnings:
                            self.progress_queue.put(("log", f"    -> [Cảnh báo] Trang {page_num}: {', '.join(quality.warnings)}\n"))
                        if not quality.passed:
                            initial_md, initial_quality = page_md, quality
                            self.progress_queue.put(("log", f"    -> Quality retry trang {page_num}: {', '.join(quality.errors)}\n"))
                            try:
                                retry_md = ocr_qwen_images(
                                    self.client,
                                    hybrid_model if is_hybrid else resolve_qwen_model(self.model_name),
                                    [img_path],
                                    has_table=has_table,
                                    extra_instruction="\n\n" + QUALITY_RETRY_INSTRUCTION.format(errors=", ".join(quality.errors)),
                                    log_func=lambda message: self.progress_queue.put(("log", f"    -> [Trang {page_num}] {message}\n")),
                                    before_request=before_qwen_request,
                                )
                            except PipelineCancelled:
                                return None
                            except Exception as retry_error:
                                page_md, quality = initial_md, initial_quality
                                self.progress_queue.put(("log", f"    -> [Cảnh báo] Retry trang {page_num} bị lỗi; giữ kết quả ban đầu: {retry_error}\n"))
                            else:
                                retry_md = apply_page_assets(retry_md, page_num, extracted_paths)
                                retry_quality = evaluate_page(retry_md, self.output_dir)
                                if retry_quality.warnings:
                                    self.progress_queue.put(("log", f"    -> [Cảnh báo] Retry trang {page_num}: {', '.join(retry_quality.warnings)}\n"))
                                page_md, quality = choose_best_page(
                                    initial_md, initial_quality, retry_md, retry_quality
                                )
                            if not quality.passed:
                                self.progress_queue.put((
                                    "log",
                                    f"    -> [Cảnh báo] Trang {page_num} vẫn chưa đạt quality gate "
                                    f"({', '.join(quality.errors)}); dùng kết quả tốt nhất và tiếp tục.\n",
                                ))

                        qwen_seconds = time.perf_counter() - qwen_started_at
                        elapsed = time.perf_counter() - started_at
                        self.progress_queue.put(("log", f"    Render: {render_timings[idx]:.1f}s | Paddle: {paddle_seconds:.1f}s | Qwen: {qwen_seconds:.1f}s\n"))
                        self.progress_queue.put(("log", f"    -> Hoàn thành Trang {page_num} ({elapsed:.1f} giây)\n"))
                        
                        with self.progress_lock:
                            self.pages_processed = getattr(self, "pages_processed", 0) + 1
                            pages_done = self.pages_processed
                            if pages_done % 5 == 0:
                                import gc
                                gc.collect()
                                self.progress_queue.put(("log", f"    -> [Memory] Đã dọn rác RAM sau {pages_done} trang.\n"))
                            
                            elapsed_total = time.perf_counter() - pdf_start_time
                            avg_time = elapsed_total / pages_done
                            remaining_pages = total_pages - pages_done
                            eta_seconds = avg_time * remaining_pages
                            self.progress_queue.put(("stats_update", (avg_time, eta_seconds)))
                        
                        return (page_num, page_md)

                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                        futures = []
                        for idx, img_path in enumerate(page_images):
                            future = executor.submit(process_page_worker, (idx, img_path))
                            # Always stop the live timer, including cancellation and
                            # exceptions raised by a failed quality gate.
                            future.add_done_callback(
                                lambda _future, page_num=idx + 1:
                                self.progress_queue.put(("page_timer_end", page_num))
                            )
                            futures.append(future)
                        
                        results = []
                        for future in concurrent.futures.as_completed(futures):
                            res = future.result()
                            if res is not None:
                                results.append(res)
                    
                    results.sort(key=lambda x: x[0])
                    from app.core.quality_gate import validate_page_numbers
                    page_quality = validate_page_numbers([page_num for page_num, _ in results], total_pages)
                    if not page_quality.passed and not self.stop_event.is_set():
                        raise RuntimeError("Thiếu trang trước khi lưu: " + ", ".join(page_quality.errors))
                    
                    for page_num, page_md in results:
                        ocr_contents.append(f"<!-- Page {page_num} -->\n\n{page_md}")
                    
                    if self.stop_event.is_set():
                        break
                        
                    # Save results
                    final_md = "\n\n".join(ocr_contents) + "\n"
                    try:
                        final_md = finalize_markdown(final_md)
                    except Exception as pp_err:
                        self.progress_queue.put(("log", f"    -> [Chú ý] Không thể hoàn thiện Markdown: {pp_err}\n"))
                    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
                    temp_output_path.write_text(final_md, encoding="utf-8")
                    os.replace(temp_output_path, output_path)
                    self.progress_queue.put(("log", f"  - Đã lưu kết quả tại: {output_path.name}\n"))
                    file_elapsed = time.perf_counter() - pdf_start_time
                    self.progress_queue.put(("log", f"  - Tổng thời gian OCR file: {format_elapsed(file_elapsed)}\n"))
            
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
            response = Client(host="http://localhost:11434").list()
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
        with Image.open(self.latest_overlay_path) as source:
            image = source.convert("RGB")
            image.thumbnail((1000, 720))
            photo = ImageTk.PhotoImage(image)
        canvas.create_image(10, 10, anchor=tk.NW, image=photo)
        canvas.create_text(10, image.height + 25, anchor=tk.W, fill="white",
                           text="Xanh dương: vùng text | Cam: ranh giới cột | Đỏ: vùng bảng")
        window._overlay_image = photo

    def update_inline_layout_overlay(self, overlay_path: Path):
        """Render the latest diagnostic image inside the main GUI."""
        try:
            with Image.open(overlay_path) as source:
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
        worker = OCRWorker(target_path, Path(output), self.progress_queue, self.stop_event, self.resume_event, model_name, workers=self.workers_var.get() if hasattr(self, 'workers_var') else 1)
        self.worker_thread = threading.Thread(target=worker.run, daemon=True)
        self.worker_thread.start()


    def toggle_pause(self):
        if not self._is_running: return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.resume_event.clear()
            self.btn_pause.configure(text="Tiếp tục")
            self.progress_queue.put(("log", "\n[TẠM DỪNG] Tiến trình đang tạm dừng...\n"))
        else:
            self.resume_event.set()
            self.btn_pause.configure(text="Tạm dừng")
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
                    self.file_progress_var.set(f"File {curr}/{total}: {name}")
                    percent = (curr / total) * 100
                    self.file_progressbar["value"] = percent
                elif msg_type == "page_progress":
                    curr, total = data
                    self.page_progress_var.set(f"Trang {curr}/{total}")
                    percent = (curr / total) * 100
                    self.page_progressbar["value"] = percent
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
        if active_timers and self._is_running:
            now = time.perf_counter()
            running = " | ".join(
                f"Trang {page_num}: {now - started_at:.1f}s"
                for page_num, started_at in sorted(active_timers.items())
            )
            self.stats_active_var.set(f"Đang OCR: {running}")
        elif hasattr(self, "stats_active_var") and self._is_running:
            self.stats_active_var.set("Đang OCR: đang chuẩn bị...")
        self.root.after(100, self.poll_queue)

def main():
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
