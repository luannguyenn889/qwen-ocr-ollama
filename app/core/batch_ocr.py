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
import time
import gc

def generate_with_retry(client, kwargs, max_retries=3, log_func=print):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.generate(**kwargs)
        except Exception as e:
            last_err = e
            log_func(f"    -> [Warning] Ollama generate failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                gc.collect()
    raise last_err

# pyrefly: ignore [missing-import]
import pymupdf  # PyMuPDF
# pyrefly: ignore [missing-import]
from ollama import Client
# pyrefly: ignore [missing-import]
from PIL import Image
from io import BytesIO

MODEL = "qwen3.5:4b"


class PipelineCancelled(Exception):
    """Raised internally when a frontend cancels a shared pipeline operation."""

BoundingBox = tuple[float, float, float, float]

# Đặt False để bỏ qua Paddle layout detection và buộc gửi cả trang đầy đủ cho Qwen
ENABLE_LAYOUT_DETECTION = True

# DPI độ phân giải cao hơn dành riêng cho các trang có bảng biểu được nhận diện
TABLE_RENDER_DPI = 300

# Hàm đối chiếu/ánh xạ tên mô hình từ GUI sang tên mô hình Ollama tương ứng
def resolve_qwen_model(model_name: str) -> str:
    """Ánh xạ nhãn mô hình cũ trong GUI sang định danh mô hình thực tế của Ollama."""
    selected = model_name.strip()
    if selected.casefold() == "hybrid":
        return MODEL
    if selected.casefold().startswith("hybrid ("):
        match = re.search(r"\+\s*([^()]+)\)", selected)
        return match.group(1).strip() if match else MODEL
    if selected.casefold().startswith("paddleocr ("):
        # Paddle hiện tại chỉ dùng để nhận diện layout; giữ lựa chọn Qwen hợp lệ làm mặc định.
        return MODEL
    if selected.casefold().startswith("hybrid:"):
        return selected.partition(":")[2].strip() or MODEL
    return selected or MODEL

# Hướng dẫn prompt chính để nạp cho mô hình Vision
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

# Chỉ lệnh bắt buộc đối với trang chứa bảng biểu
TABLE_SAFE_INSTRUCTION = """

LƯU Ý BẮT BUỘC VỀ BẢNG: Trang này có bảng biểu. Đọc toàn bộ ảnh trang, không đọc theo cột bị cắt.
MỌI bảng trên trang này bắt buộc dùng HTML `<table>` với `<tr>`, `<th>`, `<td>`, `rowspan` và `colspan`
khi cần. TUYỆT ĐỐI không dùng bảng Markdown bằng ký tự `|`. Không biến nội dung các ô thành bullet lồng
nhau, không lặp tiêu chí giữa các hàng, không tự suy diễn hoặc hoàn thiện chữ không nhìn rõ.
""".strip()

# Hướng dẫn phục hồi cấu trúc bảng HTML trong trường hợp thử lại
TABLE_HTML_RETRY_INSTRUCTION = """
NHIỆM VỤ CHỈ ĐỂ KHÔI PHỤC CẤU TRÚC BẢNG TỪ ẢNH: Trả về nội dung tài liệu và các bảng dưới dạng HTML hợp lệ.
Mỗi bảng phải có đủ thẻ mở/đóng: `<table>...</table>`. Mỗi hàng phải là `<tr>...</tr>` và mỗi ô là
`<td>...</td>` hoặc `<th>...</th>`. Dùng `<br>` chỉ bên trong một ô. Không bao giờ xuất dòng chứa các cột
phân tách bằng `|`; không xuất văn bản bảng rời ngoài `<table>`. Giữ nguyên nội dung nhìn thấy, không suy diễn.
""".strip()

# Prompt hướng dẫn sửa chữa cấu trúc bảng bị hỏng
TABLE_STRUCTURE_REPAIR_PROMPT = """Bạn là bộ sửa cấu trúc bảng OCR.
Đọc ảnh trang gốc để kiểm chứng. Kết quả OCR trước đó có bảng bị vỡ thành các dòng chứa dấu `|` rời.
Hãy trả về lại TOÀN BỘ nội dung trang dưới dạng Markdown, nhưng mọi bảng bắt buộc là HTML hợp lệ:
`<table><tr><th>...</th></tr><tr><td>...</td></tr></table>`.
Không được có dòng bảng dùng dấu `|`; không lời dẫn giải; không bỏ, lặp hoặc suy diễn nội dung.

KẾT QUẢ OCR CẦN SỬA CẤU TRÚC:
---
{broken_markdown}
---""".strip()

QUALITY_RETRY_INSTRUCTION = """The previous OCR result failed these quality checks: {errors}.
OCR this page again from the image. Preserve all visible content and layout. Ensure Vietnamese diacritics and spaces are correct, math delimiters are balanced, tables are valid Markdown/HTML, and do not emit unresolved placeholders."""

# Hàm kiểm tra xem trang PDF có phải là các mảnh quét nhỏ xếp kề nhau hay không
def _is_tiled_scan(page, image_list) -> bool:
    """Phát hiện các máy quét lưu trữ một trang dưới dạng nhiều phân mảnh raster xếp kề nhau."""
    if len(image_list) < 6:
        return False
    covered_area = 0.0
    for image_info in image_list:
        covered_area += sum(rect.get_area() for rect in page.get_image_rects(image_info[0]))
    return covered_area / page.rect.get_area() >= 0.75


# Hàm kiểm tra trang scan dạng mảnh từ bên ngoài gọi vào
def page_is_tiled_scan(pdf_path: Path, page_index: int) -> bool:
    """Tải trang PDF và kiểm tra xem có cấu trúc mảnh raster xếp kề nhau không."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        return _is_tiled_scan(page, page.get_images(full=True))
    finally:
        doc.close()


# Tập ký tự chứa dấu phụ tiếng Việt
VIETNAMESE_DIACRITICS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựỳýỷỹỵ")


# Hàm phát hiện các trang nhận diện bị lỗi mất dấu tiếng Việt hoặc dính chữ
def needs_vision_retry(markdown: str) -> bool:
    """Phát hiện các lỗi nhận diện lai phổ biến: đoạn văn bị dính chữ hoặc tiếng Việt bị mất dấu phụ."""
    if re.search(r"\b\w{45,}\b", markdown, re.UNICODE):
        gc.collect()
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


# Hàm phát hiện xem cấu trúc bảng có bị vỡ thành văn bản thường hoặc danh sách không
def needs_table_retry(markdown: str) -> bool:
    """Phát hiện các ảo giác về bảng biểu thường gặp để yêu cầu thử lại toàn trang."""
    nested_bullets = sum(bool(re.match(r"^\s{4,}[-*+]\s+", line)) for line in markdown.splitlines())
    has_placeholder_heading = bool(re.search(r"^#{1,6}\s+.*\.\.\.", markdown, re.MULTILINE))
    markdown_pipe_table = bool(re.search(r"^\s*\|.+\|\s*$\n\s*\|[\s:|-]+\|", markdown, re.MULTILINE))
    html = markdown.casefold()
    malformed_html_table = "<table" in html and "</table>" not in html
    
    # Kiểm tra xem có dấu hiệu của bảng trong đầu ra không
    has_table_indicators = "|" in markdown or "tr>" in html or "td>" in html or "table" in html
    
    # Chúng ta chỉ quan tâm đến bảng HTML bị thiếu nếu có xuất hiện chỉ báo bảng hoặc bullet thụt lề lớn
    missing_html_table = ("<table" not in html) and (has_table_indicators or nested_bullets >= 3) and not markdown_pipe_table
    
    return nested_bullets >= 3 or has_placeholder_heading or malformed_html_table or missing_html_table


def normalize_worker_count(workers: int, model_name: str) -> int:
    """Use the same conservative 1-2 worker range in CLI and GUI."""
    return max(1, min(int(workers), 2))


# Hàm trích xuất các hình ảnh từ PDF sử dụng PP-DocLayout hoặc PyMuPDF làm fallback
def extract_images_from_page(
    pdf_path: Path, page_index: int, output_img_dir: Path, prefix: str,
    layout_detector=None,
    page_image_path: Path | None = None,
    segments: list[BoundingBox] | None = None
) -> list[str]:
    """Trích xuất hình ảnh trang bằng PP-DocLayout khi khả dụng; nếu không sẽ lùi về dùng PyMuPDF."""
    # pyrefly: ignore [missing-import]
    from PIL import Image 
    from app.core.layout_detector import BoundingBox

    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height
    
    extracted_items: list[tuple[str, BoundingBox]] = []
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    layout_images_extracted = False
    img_width = page_width
    img_height = page_height
    
    # 1. Sử dụng PP-DocLayout nếu có thực thể layout_detector và page_image_path hợp lệ
    if layout_detector is not None and page_image_path is not None and page_image_path.exists():
        try:
            image_bytes = page_image_path.read_bytes()
            with Image.open(page_image_path) as img:
                img_width = img.width
                img_height = img.height
            
            # Phát hiện bảng biểu và ảnh bằng Paddle
            _, detected_images = layout_detector.detect_layout_tables_and_images(image_bytes)
            
            if detected_images:
                with Image.open(page_image_path) as image:
                    for img_idx, bbox in enumerate(detected_images, 1):
                        left, top, right, bottom = bbox
                        # Đảm bảo tọa độ nằm trong giới hạn của ảnh
                        left = max(0.0, min(left, float(img_width)))
                        top = max(0.0, min(top, float(img_height)))
                        right = max(left + 1.0, min(right, float(img_width)))
                        bottom = max(top + 1.0, min(bottom, float(img_height)))
                        
                        crop_path = output_img_dir / f"{prefix}_page_{page_index + 1}_layout_img_{img_idx}.png"
                        image.crop((left, top, right, bottom)).save(crop_path)
                        
                        # Chuẩn hóa tọa độ về dải 0.0 - 1.0
                        norm_bbox = (
                            left / img_width,
                            top / img_height,
                            right / img_width,
                            bottom / img_height
                        )
                        
                        extracted_items.append((f"images/{crop_path.name}", norm_bbox))
                layout_images_extracted = True
                print(f"    -> Extracted {len(detected_images)} images via PP-DocLayout on page {page_index + 1}.")
        except Exception as layout_err:
            print(f"    -> [Warning] PP-DocLayout image extraction failed, falling back to PyMuPDF: {layout_err}")
            layout_images_extracted = False

    # 2. Lùi về dùng PyMuPDF extract_image và get_drawings để lấy ảnh
    if not layout_images_extracted:
        image_list = page.get_images(full=True)
        tiled_scan = _is_tiled_scan(page, image_list)
        
        # Trích xuất ảnh raster thường
        for img_idx, img_info in enumerate(() if tiled_scan else image_list, 1):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                img_name = f"{prefix}_page_{page_index + 1}_img_{img_idx}.{image_ext}"
                img_path = output_img_dir / img_name
                img_path.write_bytes(image_bytes)
                
                # Xác định vị trí của ảnh trên trang
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    norm_bbox = (
                        r.x0 / page_width,
                        r.y0 / page_height,
                        r.x1 / page_width,
                        r.y1 / page_height
                    )
                else:
                    norm_bbox = (0.0, 0.0, 1.0, 1.0)
                    
                extracted_items.append((f"images/{img_name}", norm_bbox))
            except Exception as e:
                print(f"Error extracting image xref {xref} on page {page_index}: {e}")
                
        # Trích xuất hình vẽ vector
        try:
            drawings = page.get_drawings()
            candidate_rects = []
            for d in drawings:
                r = d["rect"]
                if r.is_empty:
                    continue
                if r.y1 < page_height * 0.12 or r.y0 > page_height * 0.88:
                    continue
                if r.width > page_width * 0.9 or r.height > page_height * 0.9:
                    continue
                candidate_rects.append(r)
                
            if candidate_rects:
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
                
                text_word_centres = [
                    pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
                    for word in page.get_text("words")
                ]
                
                for c_idx, rect in enumerate(merged, len(image_list) + 1):
                    words_in_rect = sum(rect.contains(centre) for centre in text_word_centres)
                    if words_in_rect >= 30:
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
                    
                    norm_bbox = (
                        crop_rect.x0 / page_width,
                        crop_rect.y0 / page_height,
                        crop_rect.x1 / page_width,
                        crop_rect.y1 / page_height
                    )
                    extracted_items.append((f"images/{img_name}", norm_bbox))
        except Exception as dev_err:
            print(f"Error extracting vector drawings on page {page_index}: {dev_err}")

    doc.close()
    
    if not extracted_items:
        return []
        
    # Cột được chuẩn hóa: danh sách các BoundingBox trong dải 0.0 - 1.0
    norm_segments: list[BoundingBox] = []
    if segments:
        width_divisor = float(img_width)
        height_divisor = float(img_height)
        for left, top, right, bottom in segments:
            norm_segments.append((
                left / width_divisor, top / height_divisor, 
                right / width_divisor, bottom / height_divisor
            ))
            
    # Hàm xác định chỉ số phân đoạn của ảnh dựa trên tọa độ trung tâm
    def get_segment_idx(norm_bbox: BoundingBox) -> int:
        if not norm_segments or len(norm_segments) <= 1:
            return 0
        center_x = (norm_bbox[0] + norm_bbox[2]) / 2
        center_y = (norm_bbox[1] + norm_bbox[3]) / 2
        for idx, (left, top, right, bottom) in enumerate(norm_segments):
            if left <= center_x <= right and top <= center_y <= bottom:
                return idx
        # Tìm phân đoạn gần nhất nếu không nằm chính xác trong phân đoạn nào
        closest_idx = 0
        min_dist = float('inf')
        for idx, (left, top, right, bottom) in enumerate(norm_segments):
            dist_x = min(abs(center_x - left), abs(center_x - right))
            dist_y = min(abs(center_y - top), abs(center_y - bottom))
            dist = dist_x + dist_y
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
        return closest_idx

    # Sắp xếp ảnh: theo thứ tự phân đoạn trước, sau đó từ trên xuống dưới, sau đó từ trái sang phải
    extracted_items.sort(key=lambda item: (get_segment_idx(item[1]), item[1][1], item[1][0]))
    
    return [path for path, _ in extracted_items]


# Hàm gộp các khối bảng Markdown bị phân mảnh thành một bảng thống nhất
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
                    # Loại bỏ các dòng trống/bình luận chen giữa các phần bảng
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


def _split_markdown_table_row(line: str) -> list[str]:
    """Split a pipe-table row without treating escaped/code-span pipes as cells."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    in_code = False
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            current.append(char)
            escaped = True
        elif char == "`":
            current.append(char)
            in_code = not in_code
        elif char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def _is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _malformed_table_to_html(rows: list[list[str]]) -> str:
    """Preserve every OCR cell in valid HTML when Markdown columns disagree."""
    output = ["<table>"]
    for row_index, cells in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        output.append("  <tr>")
        output.extend(f"    <{tag}>{cell}</{tag}>" for cell in cells)
        output.append("  </tr>")
    output.append("</table>")
    return "\n".join(output)


def repair_markdown_tables(markdown: str) -> str:
    """Repair pipe tables deterministically without inventing missing cell data.

    Consistent tables remain Markdown. If OCR produced a different number of
    cells between rows, the block becomes HTML, which permits irregular rows
    while preserving every value and preventing the rest of the document from
    being rendered as part of a broken Markdown table.
    """
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not (stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2):
            output.append(lines[index])
            index += 1
            continue

        block: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if not (candidate.startswith("|") and candidate.endswith("|") and candidate.count("|") >= 2):
                break
            block.append(lines[index])
            index += 1

        parsed = [_split_markdown_table_row(line) for line in block]
        separator_indexes = [i for i, cells in enumerate(parsed) if _is_markdown_separator(cells)]
        if len(parsed) >= 2 and separator_indexes == [1]:
            content_rows = [parsed[0], *parsed[2:]]
            column_counts = {len(cells) for cells in content_rows}
            separator_matches = len(parsed[1]) == len(parsed[0])
            if len(column_counts) > 1 or not separator_matches:
                output.extend(_malformed_table_to_html(content_rows).splitlines())
            else:
                output.extend(block)
        else:
            # It is not unambiguously a Markdown table; preserve it verbatim.
            output.extend(block)
    return "\n".join(output)


# Hàm liên kết các file ảnh đã trích xuất vào nội dung Markdown thay thế nhãn giữ chỗ
def link_extracted_images(markdown: str, extracted_paths: list[str]) -> tuple[str, list[str]]:
    """Resolve only image placeholders, preserving real links and image URLs."""
    # Sửa lỗi mô hình nhỏ hay quên đóng thẻ ảnh ở cuối dòng.
    markdown = re.sub(r"!\[([^\]\n]+)$", r"![\1](image_placeholder.png)", markdown, flags=re.MULTILINE)

    # Do not allocate an extracted image twice when Qwen already emitted its
    # real path in another image tag.
    available = [path for path in extracted_paths if f"]({path})" not in markdown]
    iterator = iter(available)
    used: list[str] = []

    def next_image(alt_text: str) -> str:
        try:
            path = next(iterator)
        except StopIteration:
            return ""
        used.append(path)
        return f"![{alt_text or 'Hình ảnh'}]({path})"

    placeholder_target = r"(?:[^\s)'\"]*/)?image_placeholder(?:\.[A-Za-z0-9]+)?"

    # Markdown images with a placeholder target, including paths such as
    # images/image_placeholder.png. Surplus placeholders are removed.
    markdown = re.sub(
        rf"!\[([^\]]*)\]\(\s*{placeholder_target}\s*\)",
        lambda match: next_image(match.group(1)),
        markdown,
        flags=re.IGNORECASE,
    )

    # HTML is occasionally emitted even on non-table pages.
    def replace_html_image(match: re.Match[str]) -> str:
        tag = match.group(0)
        alt_match = re.search(r"\balt\s*=\s*(['\"])(.*?)\1", tag, re.IGNORECASE)
        return next_image(alt_match.group(2) if alt_match else "Hình ảnh")

    markdown = re.sub(
        rf"<img\b[^>]*\bsrc\s*=\s*(['\"]){placeholder_target}\1[^>]*>",
        replace_html_image,
        markdown,
        flags=re.IGNORECASE,
    )

    # A bare placeholder is also valid model output. Convert it in place so
    # reading order is retained.
    markdown = re.sub(
        rf"(?<![\w/.-]){placeholder_target}(?![\w/.-])",
        lambda _match: next_image("Hình ảnh"),
        markdown,
        flags=re.IGNORECASE,
    )

    remaining = [path for path in available if path not in used]
    return markdown, remaining


def apply_page_assets(markdown: str, page_number: int, image_paths: list[str], formulas: list[str] | None = None) -> str:
    """Replace formula/image placeholders and append images the model did not place."""
    if formulas:
        iterator = iter(formulas)

        def replace_formula(match):
            try:
                return next(iterator)
            except StopIteration:
                return match.group(0)

        markdown = re.sub(r"formula_placeholder", replace_formula, markdown, flags=re.IGNORECASE)
    pending_images = [path for path in image_paths if f"]({path})" not in markdown]
    markdown, unplaced = link_extracted_images(markdown, pending_images)
    # Qwen may emit more image placeholders than the PDF extractor finds (it
    # often mistakes formulas or decorations for figures). These placeholders
    # have no valid file to link and should not trigger a costly full-page retry.
    markdown = re.sub(
        r"!\[[^\]]*\]\(\s*image_placeholder(?:\.[A-Za-z0-9]+)?\s*\)",
        "",
        markdown,
        flags=re.IGNORECASE,
    )
    markdown = re.sub(
        r"(?<![\w/.-])image_placeholder(?:\.[A-Za-z0-9]+)?(?![\w/.-])",
        "",
        markdown,
        flags=re.IGNORECASE,
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if unplaced:
        fallback = "\n\n".join(f"![Hình ảnh trang {page_number}]({path})" for path in unplaced)
        markdown = f"{markdown.rstrip()}\n\n{fallback}"
    return markdown


# Hàm dọn dẹp và chuẩn hóa văn bản Markdown nhận diện từ mô hình Vision
def clean_markdown(text: str) -> str:
    """
    Làm sạch Markdown đầu ra: loại bỏ rào chắn mã, lời thừa nhận diện của chatbot,
    chuẩn hóa thực thể khoảng trắng HTML, sửa công thức toán học bị nhận diện nhầm.
    """
    import re
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Loại bỏ các đoạn chào hỏi tự động của chatbot ở đầu văn bản
    text = re.sub(
        r"\A(?:based on (?:your|the) requirements|here (?:is|are) (?:the )?(?:converted|extracted|requested)|dưới đây là).+?(?:\r?\n){2,}",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    
    # Loại bỏ thực thể khoảng trắng HTML
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;nbsp;", " ")

    # Hàm lọc bỏ dấu đô-la bao bọc các câu văn bản thông thường chứa số
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
    
    # Sửa lỗi một khối công thức đô-la bao bọc nhiều phương án lựa chọn trắc nghiệm
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


def post_process_markdown(text: str) -> str:
    """Apply final Markdown repairs shared by CLI and GUI."""
    text = re.sub(r"&(?:nbsp|amp);", " ", text)
    processed_lines = []
    for line in text.splitlines():
        def split_merged_math(match):
            content = match.group(1)
            if re.search(r"\s+([A-Za-z0-9])[.)]\s+", content):
                parts = re.split(r"(\s+[A-Za-z0-9][.)]\s+)", content)
                return "".join(
                    f"${part.strip()}$" if index % 2 == 0 and part.strip() else part
                    for index, part in enumerate(parts)
                )
            return match.group(0)

        line = re.sub(r"(?<!\\)\$(.*?)(?<!\\)\$", split_merged_math, line)
        dollar_indices = [
            index for index, char in enumerate(line)
            if char == "$" and (index == 0 or line[index - 1] != "\\")
        ]
        if len(dollar_indices) % 2:
            stripped = line.rstrip()
            line = stripped[:-1] + "$." if stripped.endswith(".") else stripped + "$"
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


def finalize_markdown(markdown: str) -> str:
    """Canonical finalization used by every frontend."""
    return post_process_markdown(merge_markdown_tables(repair_markdown_tables(markdown)))


# Hàm lấy tổng số trang của file PDF
def pdf_page_count(pdf_path: Path) -> int:
    """Trả về tổng số trang của tệp PDF chỉ định."""
    doc = pymupdf.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


# Generator render trang PDF thành ảnh PNG theo tiến trình
def iter_render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150, render_timings: dict[Path, float] | None = None):
    """Render lần lượt từng trang PDF thành ảnh để luồng OCR có thể xử lý song song ngay lập tức."""
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


# Hàm render toàn bộ trang PDF thành danh sách ảnh PNG
def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    """Kết xuất toàn bộ trang PDF thành ảnh PNG."""
    return list(iter_render_pdf_to_images(pdf_path, output_dir, dpi=dpi))


# Hàm render trang chứa bảng biểu với độ phân giải (DPI) cao
def render_table_page(pdf_path: Path, page_index: int, output_dir: Path, dpi: int = TABLE_RENDER_DPI) -> Path:
    """Render trang có bảng với DPI cao hơn để nâng cao độ nét ảnh giúp OCR bảng chính xác hơn."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"page_{page_index + 1}_table_{dpi}dpi.png"
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc.load_page(page_index).get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        pix.save(str(output_path))
    finally:
        doc.close()
    return output_path


# Hàm xóa rác bộ nhớ cache GPU
def clear_gpu_cache():
    """Giải phóng tài nguyên bộ nhớ cache của PyTorch, PaddlePaddle và garbage collection hệ thống."""
    import gc
    gc.collect()
    
    # Xóa bộ nhớ cache CUDA của PyTorch
    try:
        # pyrefly: ignore [missing-import]
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    # Xóa bộ nhớ cache CUDA của PaddlePaddle
    try:
        # pyrefly: ignore [missing-import]
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def ocr_qwen_images(
    client, model: str, images: list[Path], *, has_table: bool = False,
    extra_instruction: str = "", log_func=print, before_request=None,
) -> str:
    """Canonical Qwen OCR and table-retry flow shared by CLI and GUI."""
    parts = []
    for image_number, image_path in enumerate(images, 1):
        if before_request is not None:
            before_request()
        instruction = ("\n\n" + TABLE_SAFE_INSTRUCTION if has_table else "") + extra_instruction
        log_func(f"Qwen vision OCR ({image_number}/{len(images)}).")
        chunks = generate_with_retry(
            client,
            dict(
                model=model, prompt=PROMPT + instruction, images=[str(image_path)],
                think=False, stream=True,
                options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
                keep_alive="10m",
            ),
            log_func=log_func,
        )
        parts.append(clean_markdown("".join(chunk.response for chunk in chunks)))

    markdown = "\n\n".join(part for part in parts if part.strip())
    if has_table and needs_table_retry(markdown):
        log_func("Table-safe retry on full page.")
        if before_request is not None:
            before_request()
        chunks = generate_with_retry(
            client,
            dict(
                model=model,
                prompt=PROMPT + "\n\n" + TABLE_SAFE_INSTRUCTION + "\n\n" + TABLE_HTML_RETRY_INSTRUCTION,
                images=[str(images[0])], think=False, stream=True,
                options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
                keep_alive="10m",
            ),
            log_func=log_func,
        )
        markdown = clean_markdown("".join(chunk.response for chunk in chunks))
        if needs_table_retry(markdown):
            log_func("Table structural repair.")
            if before_request is not None:
                before_request()
            chunks = generate_with_retry(
                client,
                dict(
                    model=model,
                    prompt=TABLE_STRUCTURE_REPAIR_PROMPT.format(broken_markdown=markdown),
                    images=[str(images[0])], think=False, stream=True,
                    options={"temperature": 0, "num_ctx": 12288, "num_predict": 6144},
                    keep_alive="10m",
                ),
                log_func=log_func,
            )
            markdown = clean_markdown("".join(chunk.response for chunk in chunks))
    return markdown


def ocr_coordinate_blocks(
    client, model: str, page_image: Path, typed_blocks, image_paths: list[str],
    output_dir: Path, *, log_func=print, before_request=None,
) -> str | None:
    """OCR a full page once and use detected image order to resolve placeholders.

    Layout blocks are deliberately not OCRed one-by-one. Dense mathematical pages
    can contain dozens of blocks, which previously caused one Ollama request per
    block and made a page take minutes.
    """
    from app.core.block_assembler import DocumentBlock

    if not isinstance(typed_blocks, list):
        return None
    blocks = [DocumentBlock(kind, tuple(bbox)) for kind, bbox in typed_blocks]
    if not image_paths or not any(block.kind == "image" for block in blocks):
        return None
    if not any(block.kind != "image" for block in blocks):
        return None

    image_count = sum(block.kind == "image" for block in blocks)
    log_func(
        f"Fast layout: {len(blocks)} blocks, {image_count} image blocks; "
        "OCR full page in one request."
    )
    placement_instruction = """

IMPORTANT IMAGE PLACEMENT: Preserve the full-page reading order. For each visible
figure or illustration, emit exactly one `![Description](image_placeholder.png)`
at its original position between the surrounding paragraphs. Do not OCR separate
layout blocks and do not move all figures to the end.
""".rstrip()
    markdown = ocr_qwen_images(
        client, model, [page_image],
        has_table=any(block.kind == "table" for block in blocks),
        extra_instruction=placement_instruction,
        log_func=log_func,
        before_request=before_request,
    )
    return apply_page_assets(markdown, 0, image_paths)


# Hàm chính xử lý OCR cho một tệp PDF đơn lẻ
def process_single_pdf(pdf_path: Path, output_dir: Path, client: Client, model_name: str, workers: int = 1):
    """
    Tiến hành lập trình tự render và nhận diện OCR toàn bộ tệp PDF:
    - Khởi tạo thư mục và quét số trang.
    - Chạy phân tích bố cục PaddleOCR để phát hiện bảng/cột.
    - Xử lý nhận diện và ghép nối nội dung.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    workers = normalize_worker_count(workers, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"
    temp_output_path = output_dir / f"{pdf_path.stem}.md.tmp"
    print(f"\nProcessing: {pdf_path.name} -> {output_path.name}")
    document_started_at = perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        total_pages = pdf_page_count(pdf_path)
        print(f"Pipelining render and OCR for {total_pages} pages (workers={workers})...")
        from app.core.formula_ocr import formula_ocr_status
        _, formula_status = formula_ocr_status()
        print(f"Formula OCR status: {formula_status}.")
        
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        render_timings: dict[Path, float] = {}
        layout_lock = threading.Lock()
        is_hybrid = "hybrid" in model_name.lower()
        hybrid_model = resolve_qwen_model(model_name)
        layout_detector = None
        if ENABLE_LAYOUT_DETECTION:
            try:
                from app.core.layout_detector import create_layout_detector
                layout_detector, layout_status = create_layout_detector()
                print(f"Layout status: {layout_status}.")
            except Exception as layout_error:
                print(f"Layout status: disabled; using full pages ({layout_error}).")

        # Hàm worker chạy nhận diện OCR cho từng trang song song
        def process_page_worker(idx_img):
            idx, img_path = idx_img
            page_num = idx + 1
            
            print(f"OCR'ing page {page_num}/{total_pages}: {img_path.name}...")
            started_at = perf_counter()

            page_md = ""
            error = None
            qwen_images = [img_path]
            has_table = False
            segments = None
            ordered_blocks = []
            render_seconds = render_timings.get(img_path, 0.0)
            paddle_seconds = 0.0
            qwen_seconds = 0.0

            # Phân tích bố cục bằng PaddleOCR nếu bộ phát hiện được nạp thành công
            if layout_detector is not None:
                try:
                    paddle_started_at = perf_counter()
                    with layout_lock:
                        segments, has_table = layout_detector.analyse(img_path)
                        ordered_blocks = layout_detector.detect_ordered_blocks(img_path.read_bytes())
                    paddle_seconds = perf_counter() - paddle_started_at
                    if segments and len(segments) >= 2:
                        from app.core.layout_detector import crop_segments
                        qwen_images = crop_segments(img_path, segments, temp_dir_path / "segments")
                        print(f"    -> Layout detected {len(qwen_images)} segments on page {page_num}.")
                    if has_table:
                        # Cắt nhỏ bảng sẽ làm hỏng ngữ cảnh hàng/cột, dùng ảnh gốc độ phân giải cao
                        qwen_images = [render_table_page(pdf_path, idx, temp_dir_path / "table_pages")]
                        print(f"    -> Layout detected table on page {page_num}; re-rendered at {TABLE_RENDER_DPI} DPI.")
                except Exception as layout_error:
                    print(f"    [Warning] Layout detection failed on page {page_num}; using full page: {layout_error}")
                    qwen_images, has_table = [img_path], False
                    segments = None
                    ordered_blocks = []

            # 1. Trích xuất hình ảnh vật lý từ trang PDF
            extracted_img_paths = []
            try:
                output_img_dir = output_dir / "images"
                extracted_img_paths = extract_images_from_page(
                    pdf_path, idx, output_img_dir, pdf_path.stem,
                    layout_detector=layout_detector,
                    page_image_path=img_path,
                    segments=segments
                )
            except Exception as img_err:
                print(f"    -> [Warning] Failed to extract images: {img_err}")

            # 1.5. Trích xuất công thức toán học từ trang PDF bằng LaTeX-OCR
            formulas_latex = []
            if layout_detector is not None and ENABLE_LAYOUT_DETECTION:
                try:
                    image_bytes = img_path.read_bytes()
                    _, _, detected_formulas = layout_detector.detect_layout_tables_images_and_formulas(image_bytes)
                    if detected_formulas:
                        def get_segment_idx(bbox):
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            if segments:
                                for seg_idx, (left, top, right, bottom) in enumerate(segments):
                                    if left <= center_x <= right and top <= center_y <= bottom:
                                        return seg_idx
                            return 0
                        
                        formulas_with_keys = []
                        for bbox in detected_formulas:
                            seg_idx = get_segment_idx(bbox)
                            formulas_with_keys.append((bbox, (seg_idx, bbox[1], bbox[0])))
                        formulas_with_keys.sort(key=lambda item: item[1])
                        
                        from app.core.formula_ocr import recognize_formula
                        
                        with Image.open(img_path) as image:
                            for idx_f, (bbox, _) in enumerate(formulas_with_keys, 1):
                                left, top, right, bottom = map(int, bbox)
                                left = max(0, left - 4)
                                top = max(0, top - 4)
                                right = min(image.width, right + 4)
                                bottom = min(image.height, bottom + 4)
                                
                                crop_img = image.crop((left, top, right, bottom))
                                buf = BytesIO()
                                crop_img.save(buf, format="PNG")
                                latex_bytes = buf.getvalue()
                                
                                latex_str = recognize_formula(latex_bytes)
                                if latex_str:
                                    height = bottom - top
                                    if height > 70:
                                        formulas_latex.append(f"\n$$\n{latex_str}\n$$\n")
                                    else:
                                        formulas_latex.append(f"${latex_str}$")
                        print(f"    -> Extracted {len(formulas_latex)} formulas via LaTeX-OCR on page {page_num}.")
                except Exception as formula_err:
                    print(f"    -> [Warning] Failed to extract formulas: {formula_err}")

            # 2. Thực hiện OCR và làm sạch kết quả bằng Qwen
            qwen_model = hybrid_model if is_hybrid else resolve_qwen_model(model_name)
            try:
                qwen_started_at = perf_counter()
                formula_instruction = (
                    "\n\nLƯU Ý CÔNG THỨC TOÁN HỌC: Hãy thay thế mọi công thức hoặc "
                    "biểu thức toán học phức tạp bằng nhãn chính xác: formula_placeholder."
                    if formulas_latex else ""
                )
                page_md = ocr_coordinate_blocks(
                    client, qwen_model, img_path, ordered_blocks, extracted_img_paths,
                    temp_dir_path / "layout_blocks" / f"page_{page_num}",
                    log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                )
                if page_md is not None:
                    print(f"    -> OCRed page {page_num} once and placed images in reading order.")
                else:
                    page_md = ocr_qwen_images(
                        client, qwen_model, qwen_images, has_table=has_table,
                        extra_instruction=formula_instruction,
                        log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                    )
                qwen_seconds = perf_counter() - qwen_started_at
            except Exception as ollama_err:
                qwen_seconds = perf_counter() - qwen_started_at
                error = f"Qwen OCR failed: {ollama_err}"

            # 2.5. Thay thế placeholder công thức và hình ảnh theo thứ tự đọc
            if extracted_img_paths:
                print(f"    -> Extracted {len(extracted_img_paths)} images from PDF page {page_num}.")
            if page_md:
                page_md = apply_page_assets(page_md, page_num, extracted_img_paths, formulas_latex)

            # 3. Quality gate: retry only this page once from the original full-page image.
            if not error:
                from app.core.quality_gate import choose_best_page, evaluate_page
                report = evaluate_page(page_md, output_dir)
                if report.warnings:
                    print(f"    -> [Warning] Page {page_num}: {', '.join(report.warnings)}")
                if not report.passed:
                    initial_md, initial_report = page_md, report
                    print(f"    -> Quality retry page {page_num}: {', '.join(report.errors)}")
                    try:
                        retry_md = ocr_qwen_images(
                            client, qwen_model, [img_path], has_table=has_table,
                            extra_instruction="\n\n" + QUALITY_RETRY_INSTRUCTION.format(errors=", ".join(report.errors)),
                            log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                        )
                        retry_md = apply_page_assets(retry_md, page_num, extracted_img_paths, formulas_latex)
                        second_report = evaluate_page(retry_md, output_dir)
                        if second_report.warnings:
                            print(f"    -> [Warning] Retry page {page_num}: {', '.join(second_report.warnings)}")
                        page_md, report = choose_best_page(
                            initial_md, initial_report, retry_md, second_report
                        )
                    except Exception as retry_error:
                        page_md, report = initial_md, initial_report
                        print(f"    -> [Warning] Quality retry failed on page {page_num}; keeping original result: {retry_error}")
                    if not report.passed:
                        print(
                            f"    -> [Warning] Page {page_num} still failed quality gate "
                            f"({', '.join(report.errors)}); using best result and continuing."
                        )

            qwen_seconds = perf_counter() - qwen_started_at if 'qwen_started_at' in locals() else qwen_seconds
            elapsed = perf_counter() - started_at
            print(f"    Render: {render_seconds:.1f}s | Paddle: {paddle_seconds:.1f}s | Qwen: {qwen_seconds:.1f}s")
            print(f"Page {page_num} done in {elapsed:.1f}s.")
            return page_num, page_md, error

        # Gửi tác vụ OCR ngay khi từng trang được render xong
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for index, image_path in enumerate(iter_render_pdf_to_images(pdf_path, temp_dir_path, dpi=150, render_timings=render_timings)):
                futures.append(executor.submit(process_page_worker, (index, image_path)))
            results = [future.result() for future in futures]
            
        # Sắp xếp lại theo đúng thứ tự số trang ban đầu
        results.sort(key=lambda x: x[0])
        from app.core.quality_gate import validate_page_numbers
        page_report = validate_page_numbers([page_num for page_num, _, _ in results], total_pages)
        if not page_report.passed:
            raise RuntimeError("OCR document quality gate failed: " + ", ".join(page_report.errors))
        failures = [(page_num, error) for page_num, _, error in results if error]
        if failures:
            details = "; ".join(f"page {page_num}: {error}" for page_num, error in failures)
            raise RuntimeError(f"OCR failed; existing output was preserved ({details})")

        ocr_contents = [f"<!-- Page {p_num} -->\n\n{p_md}" for p_num, p_md, _ in results]
            
        final_md = "\n\n".join(ocr_contents) + "\n"
        try:
            final_md = finalize_markdown(final_md)
        except Exception as merge_err:
            print(f"    -> [Warning] Failed to finalize Markdown: {merge_err}")
        temp_output_path.write_text(final_md, encoding="utf-8")
        os.replace(temp_output_path, output_path)
        print(f"Saved OCR to {output_path}")
        document_elapsed = perf_counter() - document_started_at
        average = document_elapsed / total_pages if total_pages else 0.0
        print(f"Performance: workers={workers}, total={document_elapsed:.2f}s, average={average:.2f}s/page")
        clear_gpu_cache()
        return output_path


# Hàm main điều khiển CLI của bộ OCR hàng loạt
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
        # Tự khởi tạo cấu trúc thư mục PDF nếu rỗng và sao chép dữ liệu mẫu
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

