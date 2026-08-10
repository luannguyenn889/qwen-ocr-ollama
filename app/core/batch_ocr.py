"""
Module: batch_ocr.py
Nhiệm vụ: Tìm kiếm tất cả tệp PDF trong thư mục PDF/ và thực hiện OCR hàng loạt bằng Qwen qua Ollama.
Quy trình:
  1. Quét thư mục PDF/ lấy danh sách các tệp tin PDF.
  2. Với mỗi file PDF, render thành ảnh tạm thời bằng PyMuPDF.
  3. Gửi từng trang qua mô hình Qwen để chuyển thành Markdown.
  4. Ghép nối kết quả hoàn thiện ghi vào thư mục OCR/ dưới dạng *.md tương ứng.
"""

import sys
import os
import re
import tempfile
from pathlib import Path
from time import perf_counter
# pyrefly: ignore [missing-import]
import pymupdf  # PyMuPDF
# pyrefly: ignore [missing-import]
from ollama import Client

MODEL = "qwen3.5:4b"
# Set False to skip Paddle detection/layout and send each full page to Qwen.
ENABLE_LAYOUT_DETECTION = True
TABLE_RENDER_DPI = 300


def resolve_qwen_model(model_name: str) -> str:
    """Map legacy GUI labels to the actual Ollama model identifier."""
    selected = model_name.strip()
    if selected.casefold() == "hybrid":
        return MODEL
    if selected.casefold().startswith("hybrid ("):
        match = re.search(r"\+\s*([^()]+)\)", selected)
        return match.group(1).strip() if match else MODEL
    if selected.casefold().startswith("paddleocr ("):
        # Paddle is layout-only now; preserve a usable Qwen choice.
        return MODEL
    if selected.casefold().startswith("hybrid:"):
        return selected.partition(":")[2].strip() or MODEL
    return selected or MODEL

PROMPT = """
Convert this scanned document page image into clean Markdown format.

Follow these strict structural and formatting guidelines:
1. Document Structure & Layout:
   - Identify and format headings (using appropriate #, ##, ### levels), paragraphs, blockquotes, lists (ordered and unordered), and code blocks.
   - Maintain the natural reading flow. For multi-column layouts, read column-by-column rather than spanning across columns.
   - Detect and remove noise like running page headers, footers, page numbers, and repeating watermarks to keep the main content clean.

2. Mathematical Expressions:
   - Identify all mathematical formulas, symbols, variables, subscripts, and equations.
   - Wrap inline mathematical symbols/expressions in single dollar signs (`$...$`) and block equations in double dollar signs (`$$...$$`).
   - Use standard LaTeX notation (e.g., standard symbols, Greek letters, fractions using `\frac{num}{den}`, and subscripts).
   - Ensure every formula has its own closed pair of dollar signs. Do not merge separate items, punctuation, or non-math labels inside the same dollar sign block.

3. Tables & Figures:
   - Convert simple tables into standard Markdown tables.
   - For complex tables (with merged rows/columns or nested cells), format them using clean HTML `<table>` tags.
   - For any figures, diagrams, or illustrations, represent them with a markdown image tag: `![Description of the illustration](image_placeholder.png)`.

4. General Rules:
   - Keep the original text exactly as written, preserving the language (Vietnamese, English, etc.) and spelling.
   - Do not summarize, explain, or add any introductory/concluding text.
   - Return only the raw Markdown content. Do not wrap the final output in ```markdown blocks.
""".strip()

TABLE_SAFE_INSTRUCTION = """

LƯU Ý BẮT BUỘC VỀ BẢNG: Trang này có bảng biểu. Đọc toàn bộ ảnh trang, không đọc theo cột bị cắt.
MỌI bảng trên trang này bắt buộc dùng HTML `<table>` với `<tr>`, `<th>`, `<td>`, `rowspan` và `colspan`
khi cần. TUYỆT ĐỐI không dùng bảng Markdown bằng ký tự `|`. Không biến nội dung các ô thành bullet lồng
nhau, không lặp tiêu chí giữa các hàng, không tự suy diễn hoặc hoàn thiện chữ không nhìn rõ.
""".strip()

TABLE_HTML_RETRY_INSTRUCTION = """
NHIỆM VỤ CHỈ ĐỂ KHÔI PHỤC CẤU TRÚC BẢNG TỪ ẢNH: Trả về nội dung tài liệu và các bảng dưới dạng HTML hợp lệ.
Mỗi bảng phải có đủ thẻ mở/đóng: `<table>...</table>`. Mỗi hàng phải là `<tr>...</tr>` và mỗi ô là
`<td>...</td>` hoặc `<th>...</th>`. Dùng `<br>` chỉ bên trong một ô. Không bao giờ xuất dòng chứa các cột
phân tách bằng `|`; không xuất văn bản bảng rời ngoài `<table>`. Giữ nguyên nội dung nhìn thấy, không suy diễn.
""".strip()

TABLE_STRUCTURE_REPAIR_PROMPT = """Bạn là bộ sửa cấu trúc bảng OCR.
Đọc ảnh trang gốc để kiểm chứng. Kết quả OCR trước đó có bảng bị vỡ thành các dòng chứa dấu `|` rời.
Hãy trả về lại TOÀN BỘ nội dung trang dưới dạng Markdown, nhưng mọi bảng bắt buộc là HTML hợp lệ:
`<table><tr><th>...</th></tr><tr><td>...</td></tr></table>`.
Không được có dòng bảng dùng dấu `|`; không lời dẫn giải; không bỏ, lặp hoặc suy diễn nội dung.

KẾT QUẢ OCR CẦN SỬA CẤU TRÚC:
---
{broken_markdown}
---""".strip()

def _is_tiled_scan(page, image_list) -> bool:
    """Detect scanners that store one page as many adjacent raster tiles."""
    if len(image_list) < 6:
        return False
    covered_area = 0.0
    for image_info in image_list:
        covered_area += sum(rect.get_area() for rect in page.get_image_rects(image_info[0]))
    return covered_area / page.rect.get_area() >= 0.75


def page_is_tiled_scan(pdf_path: Path, page_index: int) -> bool:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        return _is_tiled_scan(page, page.get_images(full=True))
    finally:
        doc.close()


VIETNAMESE_DIACRITICS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def needs_vision_retry(markdown: str) -> bool:
    """Detect common Hybrid failures: glued prose or Vietnamese with lost accents."""
    if re.search(r"\b\w{45,}\b", markdown, re.UNICODE):
        return True
    letters = [char for char in markdown.casefold() if char.isalpha()]
    if len(letters) < 150:
        return False
    lower = markdown.casefold()
    signals = sum(lower.count(word) for word in (" viet ", " nam ", " cua ", " va ", " la ", " nhung ", " chuong ", " trong "))
    signals += sum(lower.count(word) for word in (" việt ", " của ", " và ", " là ", " những ", " chương "))
    if signals < 3:
        return False
    accent_ratio = sum(char in VIETNAMESE_DIACRITICS for char in letters) / len(letters)
    return accent_ratio < 0.04


def needs_table_retry(markdown: str) -> bool:
    """Detect common table hallucinations that warrant a full-page retry."""
    nested_bullets = sum(bool(re.match(r"^\s{4,}[-*+]\s+", line)) for line in markdown.splitlines())
    has_placeholder_heading = bool(re.search(r"^#{1,6}\s+.*\.\.\.", markdown, re.MULTILINE))
    markdown_pipe_table = bool(re.search(r"^\s*\|.+\|\s*$\n\s*\|[\s:|-]+\|", markdown, re.MULTILINE))
    html = markdown.casefold()
    malformed_html_table = "<table" in html and "</table>" not in html
    missing_html_table = "<table" not in html
    return nested_bullets >= 3 or has_placeholder_heading or markdown_pipe_table or malformed_html_table or missing_html_table


def extract_images_from_page(pdf_path: Path, page_index: int, output_img_dir: Path, prefix: str) -> list[str]:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    image_list = page.get_images(full=True)
    tiled_scan = _is_tiled_scan(page, image_list)
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
            
            # PDF glyphs can be encoded as vector drawings.  A cluster with
            # many selectable words is page text, not an illustration.
            text_word_centres = [
                pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
                for word in page.get_text("words")
            ]
            # Render và lưu từng cụm diagram
            for c_idx, rect in enumerate(merged, len(image_list) + 1):
                words_in_rect = sum(rect.contains(centre) for centre in text_word_centres)
                if words_in_rect >= 30:
                    print(f"Skipping vector text block on page {page_index + 1} ({words_in_rect} words).")
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


def link_extracted_images(markdown: str, extracted_paths: list[str]) -> tuple[str, list[str]]:
    """Replace Qwen's common image placeholders in extraction order.

    Vision models do not consistently use ``image_placeholder.png``: they
    often invent ``image_001.png``.  Both are placeholders, never real paths.
    """
    iterator = iter(extracted_paths)
    used: list[str] = []

    def replace(match: re.Match[str]) -> str:
        try:
            path = next(iterator)
        except StopIteration:
            return match.group(0)
        used.append(path)
        return path

    linked = re.sub(r"(?<![\w/])image_(?:placeholder|\d+)\.png", replace, markdown)
    return linked, extracted_paths[len(used):]

def clean_markdown(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Vision models occasionally prepend a chat-style English/Vietnamese
    # acknowledgement.  It is never part of the scanned document.
    text = re.sub(
        r"\A(?:based on (?:your|the) requirements|here (?:is|are) (?:the )?(?:converted|extracted|requested)|dưới đây là).+?(?:\r?\n){2,}",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    
    # Clean up HTML space entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;nbsp;", " ")

    # Vision models sometimes wrap a complete prose sentence containing numbers
    # in math delimiters. KaTeX then italicises it and collapses normal spaces.
    def unwrap_prose_math(match):
        content = match.group(1)
        words = content.split()
        math_operators = len(re.findall(r"[=+*/^_{}\\]", content))
        if len(words) >= 6 and math_operators <= 2:
            return content
        return match.group(0)

    text = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", unwrap_prose_math, text)
    cleaned_lines = []
    for line in text.splitlines():
        prose = line.strip().strip("$").strip()
        if line.count("$") % 2 and len(prose.split()) >= 6 and len(re.findall(r"[=+*/^_{}\\]", prose)) <= 2:
            line = line.replace("$", "")
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    
    # Fix math block spanning multiple options (e.g. A. $... B. ...$)
    def repl(match):
        content = match.group(1)
        parts = re.split(r"(\s+[B-D]\.\s+)", content)
        if len(parts) > 1:
            new_parts = []
            for part in parts:
                if re.match(r"^\s+[B-D]\.\s+$", part):
                    new_parts.append(part)
                else:
                    stripped = part.strip()
                    if stripped:
                        new_parts.append(f"${stripped}$")
                    else:
                        new_parts.append(part)
            return "".join(new_parts)
        return match.group(0)

    text = re.sub(r"\$([^$\n]+)\$", repl, text)
    return text

def pdf_page_count(pdf_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def iter_render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150, render_timings: dict[Path, float] | None = None):
    """Yield each rendered page immediately so OCR can overlap later renders."""
    doc = pymupdf.open(pdf_path)
    try:
        for index in range(len(doc)):
            render_started_at = perf_counter()
            page = doc.load_page(index)
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            image_name = f"page_{index + 1}.png"
            image_path = output_dir / image_name
            pix.save(str(image_path))
            if render_timings is not None:
                render_timings[image_path] = perf_counter() - render_started_at
            yield image_path
    finally:
        doc.close()


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    """Render all pages; retained for callers that require a complete list."""
    return list(iter_render_pdf_to_images(pdf_path, output_dir, dpi=dpi))


def render_table_page(pdf_path: Path, page_index: int, output_dir: Path, dpi: int = TABLE_RENDER_DPI) -> Path:
    """Render a table page at higher resolution without changing normal-page speed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"page_{page_index + 1}_table_{dpi}dpi.png"
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc.load_page(page_index).get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        pix.save(str(output_path))
    finally:
        doc.close()
    return output_path


def process_single_pdf(pdf_path: Path, output_dir: Path, client: Client, model_name: str, workers: int = 1):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"
    temp_output_path = output_dir / f"{pdf_path.stem}.md.tmp"
    print(f"\nProcessing: {pdf_path.name} -> {output_path.name}")
    document_started_at = perf_counter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        total_pages = pdf_page_count(pdf_path)
        print(f"Pipelining render and OCR for {total_pages} pages (workers={workers})...")
        
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        render_timings: dict[Path, float] = {}
        layout_lock = threading.Lock()
        is_hybrid = "hybrid" in model_name.lower()
        hybrid_model = resolve_qwen_model(model_name)
        layout_detector = None
        if ENABLE_LAYOUT_DETECTION:
            try:
                from app.core.layout_detector import LayoutDetector
                layout_detector = LayoutDetector()
                print("Paddle layout detection loaded (detection only; no text recognition).")
            except Exception as layout_error:
                print(f"[Warning] Layout detection unavailable; using full pages: {layout_error}")

        def process_page_worker(idx_img):
            idx, img_path = idx_img
            page_num = idx + 1
            print(f"OCR'ing page {page_num}/{total_pages}: {img_path.name}...")
            started_at = perf_counter()
            page_md = ""
            error = None
            qwen_images = [img_path]
            has_table = False
            render_seconds = render_timings.get(img_path, 0.0)
            paddle_seconds = 0.0
            qwen_seconds = 0.0

            # Paddle contributes only geometry.  Any failure preserves the old
            # whole-page Qwen flow for this page.
            if layout_detector is not None:
                try:
                    paddle_started_at = perf_counter()
                    with layout_lock:
                        columns, has_table = layout_detector.analyse(img_path)
                    paddle_seconds = perf_counter() - paddle_started_at
                    if len(columns) >= 2:
                        from app.core.layout_detector import crop_columns
                        qwen_images = crop_columns(img_path, columns, temp_dir_path / "columns")
                        print(f"    -> Layout detected {len(qwen_images)} columns on page {page_num}.")
                    if has_table:
                        # Cropping a merged table destroys row/column context.
                        qwen_images = [render_table_page(pdf_path, idx, temp_dir_path / "table_pages")]
                        print(f"    -> Layout detected table on page {page_num}; re-rendered at {TABLE_RENDER_DPI} DPI.")
                except Exception as layout_error:
                    print(f"    [Warning] Layout detection failed on page {page_num}; using full page: {layout_error}")
                    qwen_images, has_table = [img_path], False

            # 1. Trích xuất hình ảnh vật lý từ trang PDF trước
            extracted_img_paths = []
            try:
                output_img_dir = output_dir / "images"
                extracted_img_paths = extract_images_from_page(pdf_path, idx, output_img_dir, pdf_path.stem)
            except Exception as img_err:
                print(f"    -> [Warning] Failed to extract images: {img_err}")

            # 2. Thực hiện OCR và làm sạch kết quả
            qwen_model = hybrid_model if is_hybrid else resolve_qwen_model(model_name)
            try:
                qwen_started_at = perf_counter()
                page_parts = []
                for column_number, qwen_image in enumerate(qwen_images, 1):
                    img_instruction = ""
                    if has_table:
                        img_instruction += "\n\n" + TABLE_SAFE_INSTRUCTION
                    if extracted_img_paths and column_number == 1:
                        img_instruction += f"\n\nTrang có {len(extracted_img_paths)} hình ảnh/sơ đồ; chèn đúng số thẻ ![Mô tả](image_placeholder.png) vào vị trí phù hợp."
                    print(f"    -> Qwen vision OCR ({column_number}/{len(qwen_images)}) on page {page_num}.")
                    stream_chunks = client.generate(
                        model=qwen_model, prompt=PROMPT + img_instruction,
                        images=[str(qwen_image)], think=False, stream=True,
                        options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096}, keep_alive="10m",
                    )
                    page_parts.append(clean_markdown("".join(chunk.response for chunk in stream_chunks)))
                page_md = "\n\n".join(part for part in page_parts if part.strip())
                if has_table and needs_table_retry(page_md):
                    print(f"    -> Table-safe retry on full page {page_num}.")
                    retry_chunks = client.generate(
                        model=qwen_model, prompt=PROMPT + "\n\n" + TABLE_SAFE_INSTRUCTION + "\n\n" + TABLE_HTML_RETRY_INSTRUCTION,
                        images=[str(qwen_images[0])], think=False, stream=True,
                        options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096}, keep_alive="10m",
                    )
                    page_md = clean_markdown("".join(chunk.response for chunk in retry_chunks))
                    if needs_table_retry(page_md):
                        print(f"    -> Table structural repair on page {page_num}.")
                        repair_chunks = client.generate(
                            model=qwen_model,
                            prompt=TABLE_STRUCTURE_REPAIR_PROMPT.format(broken_markdown=page_md),
                            images=[str(qwen_images[0])], think=False, stream=True,
                            options={"temperature": 0, "num_ctx": 12288, "num_predict": 6144}, keep_alive="10m",
                        )
                        page_md = clean_markdown("".join(chunk.response for chunk in repair_chunks))
                qwen_seconds = perf_counter() - qwen_started_at
            except Exception as ollama_err:
                qwen_seconds = perf_counter() - qwen_started_at
                error = f"Qwen OCR failed: {ollama_err}"

            # 3. Thay thế placeholders bằng ảnh thật đã trích xuất
            if extracted_img_paths and page_md:
                print(f"    -> Extracted {len(extracted_img_paths)} images from PDF page {page_num}.")
                page_md, unplaced_paths = link_extracted_images(page_md, extracted_img_paths)
                if unplaced_paths:
                    fallback_images = "\n\n".join(
                        f"![Hình ảnh trang {page_num}]({path})" for path in unplaced_paths
                    )
                    page_md = f"{page_md.rstrip()}\n\n{fallback_images}"

            elapsed = perf_counter() - started_at
            print(f"    Render: {render_seconds:.1f}s | Paddle: {paddle_seconds:.1f}s | Qwen: {qwen_seconds:.1f}s")
            print(f"Page {page_num} done in {elapsed:.1f}s.")
            return page_num, page_md, error

        # Submit OCR as soon as each page is rendered instead of waiting for the full PDF.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for index, image_path in enumerate(iter_render_pdf_to_images(pdf_path, temp_dir_path, dpi=150, render_timings=render_timings)):
                futures.append(executor.submit(process_page_worker, (index, image_path)))
            results = [future.result() for future in futures]
            
        # Sắp xếp lại theo đúng thứ tự số trang ban đầu
        results.sort(key=lambda x: x[0])
        failures = [(page_num, error) for page_num, _, error in results if error]
        if failures:
            details = "; ".join(f"page {page_num}: {error}" for page_num, error in failures)
            raise RuntimeError(f"OCR failed; existing output was preserved ({details})")

        ocr_contents = [f"<!-- Page {p_num} -->\n\n{p_md}" for p_num, p_md, _ in results]
            
        final_md = "\n\n".join(ocr_contents) + "\n"
        try:
            final_md = merge_markdown_tables(final_md)
        except Exception as merge_err:
            print(f"    -> [Warning] Failed to merge tables: {merge_err}")
        temp_output_path.write_text(final_md, encoding="utf-8")
        os.replace(temp_output_path, output_path)
        print(f"Saved OCR to {output_path}")
        document_elapsed = perf_counter() - document_started_at
        average = document_elapsed / total_pages if total_pages else 0.0
        print(f"Performance: workers={workers}, total={document_elapsed:.2f}s, average={average:.2f}s/page")
        return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch OCR PDFs using Qwen through Ollama or PaddleOCR")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Name of the Ollama model to use or 'paddleocr' (default: {MODEL})")
    parser.add_argument("--input", type=str, default=None, help="Path to input PDF file or folder containing PDFs")
    parser.add_argument("--output", type=str, default=None, help="Path to output directory")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1, help="Concurrent OCR requests; benchmark 1 vs 2 on your GPU (default: 1)")
    args = parser.parse_args()

    input_path = args.input if args.input else "PDF"
    output_path = args.output if args.output else "OCR"
    
    target_path = Path(input_path).resolve()
    ocr_dir = Path(output_path).resolve()
    ocr_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = []
    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        pdf_files.append(target_path)
    elif target_path.is_dir():
        pdf_files = sorted(list(target_path.glob("*.pdf")))
    else:
        # Backward compatibility setup for empty default PDF/ directory
        target_path.mkdir(parents=True, exist_ok=True)
        pdf_files = list(target_path.glob("*.pdf"))
        if not pdf_files:
            print(f"Directory {input_path} is empty. Copying sample PDFs for demonstration...")
            sample_source = Path("samples/pdfs/test.pdf")
            if sample_source.is_file():
                import shutil
                shutil.copy(sample_source, target_path / "A.pdf")
                shutil.copy(sample_source, target_path / "B.pdf")
                print("Copied sample PDFs into input directory.")
                pdf_files = list(target_path.glob("*.pdf"))
            else:
                print("No sample PDFs found. Please place PDF files in the input directory.")
                sys.exit(0)
            
    print(f"Found {len(pdf_files)} PDF files to process.")
    print(f"Using model: {args.model}")
    client = Client(host="http://localhost:11434", timeout=60.0)
    
    total_start = perf_counter()
    for pdf_file in pdf_files:
        process_single_pdf(pdf_file, ocr_dir, client, args.model, workers=args.workers)
        
    total_elapsed = perf_counter() - total_start
    print(f"\nBatch OCR processing completed in {total_elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
