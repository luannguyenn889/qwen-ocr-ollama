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
import hashlib
import json
import unicodedata
import threading

def generate_with_retry(client, kwargs, max_retries=3, log_func=print):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.generate(**kwargs)
        except Exception as e:
            last_err = e
            log_func(f"    -> [Warning] Ollama generate failed (attempt {attempt}/{max_retries}): {e}")
            error_text = str(e).casefold()
            if "exceed_context_size_error" in error_text or "exceeds the available context size" in error_text:
                break
            if attempt < max_retries:
                time.sleep(2)
                gc.collect()
    # pyrefly: ignore [bad-raise]
    raise last_err

# pyrefly: ignore [missing-import]
import pymupdf  # PyMuPDF
# pyrefly: ignore [missing-import]
from ollama import Client
# pyrefly: ignore [missing-import]
from PIL import Image
from io import BytesIO

MODEL = "qwen3.5:4b"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 300.0


class PipelineCancelled(Exception):
    """Raised internally when a frontend cancels a shared pipeline operation."""

BoundingBox = tuple[float, float, float, float]

# Đặt False để bỏ qua Paddle layout detection và buộc gửi cả trang đầy đủ cho Qwen
ENABLE_LAYOUT_DETECTION = True

# Persist genuine content images beside Markdown. Temporary review/layout
# crops remain ephemeral; signature/stamp regions are filtered separately.
TEXT_ONLY_OUTPUT = False

# DPI độ phân giải cao hơn dành riêng cho các trang có bảng biểu được nhận diện
TABLE_RENDER_DPI = 300

# One canonical target pattern shared by linking, page cleanup and finalization.
IMAGE_PLACEHOLDER_TARGET_RE = r"(?:[^\s)'\"]*/)?image_placeholder(?:\.[A-Za-z0-9]+)?"

_orientation_classifier = None
_orientation_classifier_lock = threading.Lock()


def _orientation_result_data(result):
    """Normalize PaddleOCR result objects across PaddleOCR 3.x releases."""
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "res"):
        data = result.res
    elif hasattr(result, "to_dict"):
        data = result.to_dict()
    elif hasattr(result, "json"):
        data = result.json
    else:
        data = dict(result)
    return data.get("res", data)


def detect_text_orientation(image: Image.Image) -> int:
    """Return the corrective document rotation (0, 90, 180 or 270 degrees).

    Paddle's document-orientation classifier is loaded lazily.  OCR must remain
    usable on installations where that optional model is unavailable, so a
    classifier/download failure safely leaves the page unchanged.
    """
    global _orientation_classifier
    try:
        with _orientation_classifier_lock:
            if _orientation_classifier is None:
                # pyrefly: ignore [missing-import]
                from paddleocr import DocImgOrientationClassification
                _orientation_classifier = DocImgOrientationClassification(
                    model_name="PP-LCNet_x1_0_doc_ori"
                )

            # Paddle accepts numpy arrays and returns one result for one image.
            # pyrefly: ignore [missing-import]
            import numpy as np
            results = list(_orientation_classifier.predict(np.asarray(image.convert("RGB"))))
    except Exception as error:
        print(f"    [Warning] Auto-rotate unavailable; keeping original orientation: {error}")
        return 0

    if not results:
        return 0
    data = _orientation_result_data(results[0])
    labels = data.get("label_names", data.get("labels", data.get("label_name", [])))
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    for label in labels:
        match = re.search(r"(?:^|\D)(0|90|180|270)(?:\D|$)", str(label))
        if match:
            return int(match.group(1))
    return 0


def detect_and_rotate_page(image_path: Path) -> Image.Image:
    """Load a rendered page and rotate it to its natural reading direction."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    angle = detect_text_orientation(image)
    if angle in (90, 180, 270):
        image = image.rotate(angle, expand=True)
    return image


def orient_page_file(image_path: Path) -> int:
    """Auto-orient a temporary page file in place and return its rotation."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    angle = detect_text_orientation(image)
    if angle in (90, 180, 270):
        image = image.rotate(angle, expand=True)
    image.save(image_path, format="PNG")
    image.close()
    return angle


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

ABSOLUTE OCR-ONLY RULE (highest priority): Transcribe only text that is visibly
present in the image. Never answer questions, solve exercises, write essays,
continue incomplete passages, infer an answer key, summarize, explain, or add
any new content. If the page is an exam or questionnaire, copy the questions
and blank answer areas only. Output answers or solutions only when those exact
words are visibly printed on the current image. When text is unclear, preserve
the readable portion; do not guess or complete it from context.

Follow these strict structural and formatting guidelines:
1. Document Structure & Layout:
   - Identify and format headings (using appropriate #, ##, ### levels), paragraphs, blockquotes, lists (ordered and unordered), and code blocks.
   - Put each heading on its own physical line. Never place two Markdown headings (`#`, `##`, `###`, etc.) on the same line.
   - Put each multiple-choice option (A., B., C., D. or A), B), C), D)) on its own line; never join two options on one line and never use `\\hfill`.
   - For official/administrative documents with two-column headers (Issuing Agency on the left and National Motto/Date on the right), transcribe BOTH columns completely on separate lines.
   - Maintain the natural reading flow. For multi-column layouts, read column-by-column rather than spanning across columns.
   - Detect and remove noise like running page headers, footers, page numbers, repeating publication names, and footer hashtags to keep the main content clean.

2. Mathematical Expressions:
   - Identify all mathematical formulas, symbols, variables, subscripts, and equations.
   - Wrap inline mathematical symbols/expressions in single dollar signs (`$...$`) and block equations in double dollar signs (`$$...$$`).
   - Use standard LaTeX notation (e.g., standard symbols, Greek letters, fractions using `\\frac{num}{den}`, and subscripts).
   - Ensure every formula has its own closed pair of dollar signs. Do not merge separate items, punctuation, or non-math labels inside the same dollar sign block.

3. Tables & Figures:
   - Convert simple tables into standard Markdown tables.
   - For complex tables (with merged rows/columns or nested cells), format them using clean HTML `<table>` tags.
   - If a table continues from the previous page, preserve its column structure and do not invent a new header. Keep image placeholders outside table markup.
   - Use `image_placeholder` only for genuine photographs, charts, logos, or primarily graphical illustrations.
   - Text boxes, callouts, framed notes, forms, and flowchart/diagram nodes containing text MUST be transcribed completely as Markdown (use blockquotes, lists, or tables where appropriate). Never replace their readable text with an image placeholder.
   - For a genuine figure that also contains readable labels, emit one image tag and transcribe its important visible text immediately below the tag.
   - Layout labels are only hints. Never let an `image`, `figure`, `seal`, stamp,
     signature, drawing, or overlapping graphic region suppress readable text.
     Transcribe every readable linguistic text element first, regardless of the
     document type or where it appears (including text beside or underneath a
     stamp/signature and text printed inside a graphic).
   - Do not turn non-text pen strokes, signature flourishes, or stamp artwork
     into invented words. Transcribe handwriting only when it is genuinely
     readable as text. Transcribe text inside a stamp only when the characters
     are clear enough to read without guessing.
   - Do not emit an image tag for a signature or stamp by default. Do not add an
     evidence note such as `[Signed and stamped]` unless a separate instruction
     explicitly requests it.

4. General Rules:
   - Keep the original text exactly as written, preserving the language (Vietnamese, English, etc.) and spelling.
   - Inspect stylized/display fonts character by character, especially Vietnamese diacritics and easily confused letters. Use surrounding context only to choose among glyphs that are actually visible, never to invent text.
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
CHỈ chép chữ thực sự nhìn thấy trong ảnh. TUYỆT ĐỐI KHÔNG trả lời câu hỏi, giải bài, viết bài văn,
hoàn thành nội dung còn thiếu hoặc thêm đáp án không hiện diện trong ảnh.

KẾT QUẢ OCR CẦN SỬA CẤU TRÚC:
---
{broken_markdown}
---""".strip()

QUALITY_RETRY_INSTRUCTION = """The previous OCR result failed these quality checks: {errors}.
OCR this page again from the image. Preserve all visible content and layout. Ensure Vietnamese diacritics and spaces are correct, math delimiters are balanced, tables are valid Markdown/HTML, and do not emit unresolved placeholders. Transcribe only text visibly present in the image; never answer questions, solve exercises, write essays, or add inferred content."""

MIXED_VI_EN_RETRY_INSTRUCTION = """This is a mixed Vietnamese-English document. Detect language per sentence and term.
Preserve English words, abbreviations, product names, URLs and code exactly without adding Vietnamese diacritics.
For Vietnamese text, copy every visible diacritic character by character. Do not translate, localize, or autocorrect either language."""

REPETITION_RETRY_INSTRUCTION = """The previous output entered a repetition loop. Transcribe each visible line exactly once.
Stop when reaching the physical bottom of the image. Do not continue, reconstruct, or repeat a paragraph even when the background, watermark, or decorative typography is ambiguous."""


def quality_retry_instruction(markdown: str, errors: str) -> str:
    """Build a document-type-specific retry instruction."""
    from app.core.quality_gate import detect_language_profile
    instruction = QUALITY_RETRY_INSTRUCTION.format(errors=errors)
    if detect_language_profile(markdown) == "mixed_vi_en":
        instruction += "\n\n" + MIXED_VI_EN_RETRY_INSTRUCTION
    if "repetition_loop" in errors or "repeated_words" in errors:
        instruction += "\n\n" + REPETITION_RETRY_INSTRUCTION
    return instruction

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
    """Request vision retry only for high-confidence quality failures."""
    from app.core.quality_gate import evaluate_page
    retry_errors = {"glued_words", "missing_vietnamese_diacritics"}
    return any(error.partition(":")[0] in retry_errors for error in evaluate_page(markdown, ".").errors)


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

    from app.core.quality_gate import html_table_structure_errors
    invalid_spans = bool(html_table_structure_errors(markdown))
    return nested_bullets >= 3 or has_placeholder_heading or malformed_html_table or missing_html_table or invalid_spans


def normalize_worker_count(workers: int, model_name: str) -> int:
    """Use the same conservative 1-2 worker range in CLI and GUI."""
    return max(1, min(int(workers), 2))


def _text_stats_in_rect(page, rect) -> tuple[int, int, float]:
    """Return word count, character count, and approximate text coverage."""
    words = page.get_text("words", clip=rect) or []
    word_count = len(words)
    character_count = sum(len(str(word[4]).strip()) for word in words if len(word) > 4)
    rect_area = max(float(rect.get_area()), 1.0)
    covered_area = 0.0
    for word in words:
        if len(word) < 5:
            continue
        word_rect = pymupdf.Rect(word[:4]) & rect
        if not word_rect.is_empty:
            covered_area += word_rect.get_area()
    return word_count, character_count, min(covered_area / rect_area, 1.0)


def _is_text_heavy_region(page, rect) -> bool:
    """Protect paragraphs and short text callouts misclassified as figures."""
    word_count, character_count, coverage = _text_stats_in_rect(page, rect)
    return (
        word_count >= 10
        or (word_count >= 4 and character_count >= 20 and coverage >= 0.035)
        or (word_count >= 2 and character_count >= 12 and coverage >= 0.12)
    )


def _padded_box(bbox: BoundingBox, width: float, height: float) -> BoundingBox:
    """Add proportional safe padding without crossing image boundaries."""
    left, top, right, bottom = bbox
    pad_x = max(10.0, (right - left) * 0.05)
    pad_y = max(10.0, (bottom - top) * 0.05)
    return (
        max(0.0, left - pad_x), max(0.0, top - pad_y),
        min(width, right + pad_x), min(height, bottom + pad_y),
    )


def _keep_vector_drawing(rect, page_width: float, page_height: float) -> bool:
    """Reject rules, borders, and marginal vector decorations before merging."""
    if rect.is_empty or rect.width <= 10 or rect.height <= 10:
        return False
    aspect_ratio = rect.width / max(rect.height, 0.001)
    if aspect_ratio > 12 or aspect_ratio < 0.08:
        return False
    if rect.y0 < page_height * 0.05 or rect.y1 > page_height * 0.95:
        return False
    if rect.width > page_width * 0.9 or rect.height > page_height * 0.9:
        return False
    return True


def _is_full_page_canvas(rect, page_width: float, page_height: float) -> bool:
    """Identify a raster used as the scanned page canvas, not an illustration."""
    page_area = max(page_width * page_height, 1.0)
    return rect.get_area() / page_area > 0.80


def _nearby_text(page, rect, margin: float = 36.0) -> str:
    expanded = pymupdf.Rect(
        max(0.0, rect.x0 - margin), max(0.0, rect.y0 - margin),
        min(page.rect.width, rect.x1 + margin), min(page.rect.height, rect.y1 + margin),
    )
    values = []
    for word in page.get_text("words") or []:
        if len(word) >= 5 and not (pymupdf.Rect(word[:4]) & expanded).is_empty:
            values.append(str(word[4]))
    return " ".join(values)


def _needs_graphic_classification(page, rect) -> bool:
    """Classify every uncaptained graphic, regardless of its page position."""
    context = _nearby_text(page, rect).casefold()
    has_caption = bool(re.search(r"\b(?:hình|ảnh|figure|photo|biểu\s*đồ|sơ\s*đồ)\s*\d*\b", context))
    return not has_caption


def _prepare_graphic_classification_image(crop_path: Path, max_edge: int) -> Path:
    """Create a bounded temporary JPEG so vision tokens stay below context limits."""
    target = crop_path.with_name(f"{crop_path.stem}.classify_{max_edge}.jpg")
    with Image.open(crop_path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=88, optimize=True)
    return target


def classify_graphic_crop(client, model: str, crop_path: Path, *, before_request=None, log_func=print):
    """Classify graphics without asking the model to transcribe their contents."""
    if before_request is not None:
        before_request()
    prompt = """Phân loại thành phần đồ họa trong ảnh crop; không OCR và không chép bất kỳ chữ nào.
Chọn đúng một category: content_image, logo, signature, stamp, decoration, uncertain.
content_image gồm ảnh chụp, biểu đồ, sơ đồ hoặc hình minh họa.
logo gồm biểu trưng cơ quan/thương hiệu, huy hiệu, logo tròn hoặc logo có chữ được thiết kế sạch.
stamp chỉ là dấu mực thực sự đóng lên tài liệu, thường có nét mực không đều, chồng lên chữ/nền giấy.
Không được chọn stamp chỉ vì logo có hình tròn hoặc tên cơ quan. Nếu không phân biệt chắc logo và con dấu,
phải chọn logo hoặc uncertain; chỉ chọn signature/stamp khi hình ảnh chứng minh rõ.
Chỉ trả JSON: {"category":"...","confidence":0.0}"""
    temporary_images: list[Path] = []

    def request(image_path: Path):
        return generate_with_retry(
            client, dict(
                model=model, prompt=prompt, images=[str(image_path)], think=False,
                stream=False, format="json",
                options={"temperature": 0, "num_ctx": 8192, "num_predict": 128},
                keep_alive="10m",
            ), log_func=log_func,
        )

    try:
        compact = _prepare_graphic_classification_image(crop_path, 1024)
        temporary_images.append(compact)
        try:
            response = request(compact)
        except Exception as first_error:
            error_text = str(first_error).casefold()
            if "exceed_context_size_error" not in error_text and "exceeds the available context size" not in error_text:
                raise
            smaller = _prepare_graphic_classification_image(crop_path, 768)
            temporary_images.append(smaller)
            log_func("Graphic classification exceeded context; retrying once with a 768px crop.")
            response = request(smaller)
        raw = getattr(response, "response", None)
        if raw is None and isinstance(response, dict):
            raw = response.get("response", "")
        data = json.loads(str(raw or ""))
        category = str(data.get("category", "uncertain")).strip().casefold()
        confidence = float(data.get("confidence", 0.0))
    except Exception as error:
        log_func(f"Graphic classification failed; kept image: {error}")
        return "uncertain", 0.0
    finally:
        for temporary_image in temporary_images:
            temporary_image.unlink(missing_ok=True)
    allowed = {"content_image", "logo", "signature", "stamp", "decoration", "uncertain"}
    return (category if category in allowed else "uncertain"), confidence


def _discard_classified_graphic(category: str, confidence: float) -> bool:
    return (
        category in {"signature", "stamp"} and confidence >= 0.98
    ) or (category == "decoration" and confidence >= 0.99)


# Hàm trích xuất các hình ảnh từ PDF sử dụng PP-DocLayout hoặc PyMuPDF làm fallback
def extract_images_from_page(
    pdf_path: Path, page_index: int, output_img_dir: Path, prefix: str,
    layout_detector=None,
    page_image_path: Path | None = None,
    segments: list[BoundingBox] | None = None,
    detected_images: list[BoundingBox] | None = None,
    table_regions: list[BoundingBox] | None = None,
    graphic_classifier=None,
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

            # Reuse the caller's single page analysis when available.
            if detected_images is None:
                _, detected_images = layout_detector.detect_layout_tables_and_images(image_bytes)

            if detected_images:
                accepted_images: list[BoundingBox] = []
                with Image.open(page_image_path) as image:
                    for img_idx, bbox in enumerate(detected_images, 1):
                        if any(_significant_bbox_overlap(bbox, table) for table in table_regions or []):
                            print(f"    -> Ignored layout image region {img_idx} overlapping an OCR table.")
                            continue
                        left, top, right, bottom = bbox
                        # Đảm bảo tọa độ nằm trong giới hạn của ảnh
                        left = max(0.0, min(left, float(img_width)))
                        top = max(0.0, min(top, float(img_height)))
                        right = max(left + 1.0, min(right, float(img_width)))
                        bottom = max(top + 1.0, min(bottom, float(img_height)))

                        pdf_rect = pymupdf.Rect(
                            left * page_width / img_width, top * page_height / img_height,
                            right * page_width / img_width, bottom * page_height / img_height,
                        )
                        if _is_text_heavy_region(page, pdf_rect):
                            print(f"    -> Kept text-heavy layout region {img_idx} for OCR instead of cropping it as an image.")
                            continue
                        left, top, right, bottom = _padded_box(
                            (left, top, right, bottom), float(img_width), float(img_height)
                        )

                        crop_path = output_img_dir / f"{prefix}_page_{page_index + 1}_layout_img_{img_idx}.png"
                        image.crop((left, top, right, bottom)).save(crop_path)
                        if _needs_graphic_classification(page, pdf_rect) and graphic_classifier is not None:
                            category, confidence = graphic_classifier(crop_path)
                            if _discard_classified_graphic(category, confidence):
                                crop_path.unlink(missing_ok=True)
                                print(f"    -> Ignored confirmed {category} layout region {img_idx}.")
                                continue

                        accepted_images.append(bbox)

                        # Chuẩn hóa tọa độ về dải 0.0 - 1.0
                        norm_bbox = (
                            left / img_width,
                            top / img_height,
                            right / img_width,
                            bottom / img_height
                        )

                        extracted_items.append((f"images/{crop_path.name}", norm_bbox))
                detected_images[:] = accepted_images
                # The detector handled the candidate set even when every item
                # was intentionally filtered. Do not re-extract rejected
                # stamps/signatures through the PyMuPDF fallback below.
                layout_images_extracted = True
                print(f"    -> Extracted {len(accepted_images)} images via PP-DocLayout on page {page_index + 1}.")
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

                # Xác định vị trí của ảnh trên trang
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    if any(_is_full_page_canvas(rect, page_width, page_height) for rect in rects):
                        print(f"    -> Ignored full-page scan canvas image {img_idx}.")
                        continue
                    # Tiny bitmap ornaments should not enter the generic image
                    # fallback at all. Keep the threshold aligned with vector
                    # crop filtering below.
                    if r.get_area() < page_width * page_height * 0.002:
                        continue
                    norm_bbox = (
                        r.x0 / page_width,
                        r.y0 / page_height,
                        r.x1 / page_width,
                        r.y1 / page_height
                    )
                    normalized_tables = [
                        (box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height)
                        for box in table_regions or []
                    ]
                    if any(_significant_bbox_overlap(norm_bbox, table) for table in normalized_tables):
                        print(f"    -> Ignored raster image {img_idx} overlapping an OCR table.")
                        continue
                else:
                    norm_bbox = (0.0, 0.0, 1.0, 1.0)

                img_name = f"{prefix}_page_{page_index + 1}_img_{img_idx}.{image_ext}"
                img_path = output_img_dir / img_name
                img_path.write_bytes(image_bytes)

                if rects and _needs_graphic_classification(page, r) and graphic_classifier is not None:
                    category, confidence = graphic_classifier(img_path)
                    if _discard_classified_graphic(category, confidence):
                        img_path.unlink(missing_ok=True)
                        print(f"    -> Ignored confirmed {category} raster image {img_idx}.")
                        continue

                extracted_items.append((f"images/{img_name}", norm_bbox))
            except Exception as e:
                print(f"Error extracting image xref {xref} on page {page_index}: {e}")

        # Trích xuất hình vẽ vector
        try:
            drawings = page.get_drawings()
            candidate_rects = []
            for d in drawings:
                r = d["rect"]
                if not _keep_vector_drawing(r, page_width, page_height):
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

                for c_idx, rect in enumerate(merged, len(image_list) + 1):
                    if _is_text_heavy_region(page, rect):
                        continue

                    padding = 10
                    crop_rect = pymupdf.Rect(
                        max(0, rect.x0 - padding),
                        max(0, rect.y0 - padding),
                        min(page_width, rect.x1 + padding),
                        min(page_height, rect.y1 + padding)
                    )
                    if crop_rect.get_area() < page_width * page_height * 0.002:
                        continue
                    zoom = 300 / 72
                    mat = pymupdf.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                    if pix.width < 40 or pix.height < 40:
                        continue

                    img_name = f"{prefix}_page_{page_index + 1}_draw_{c_idx}.png"
                    img_path = output_img_dir / img_name
                    pix.save(str(img_path))

                    if _needs_graphic_classification(page, crop_rect) and graphic_classifier is not None:
                        category, confidence = graphic_classifier(img_path)
                        if _discard_classified_graphic(category, confidence):
                            img_path.unlink(missing_ok=True)
                            print(f"    -> Ignored confirmed {category} vector region {c_idx}.")
                            continue

                    norm_bbox = (
                        crop_rect.x0 / page_width,
                        crop_rect.y0 / page_height,
                        crop_rect.x1 / page_width,
                        crop_rect.y1 / page_height
                    )
                    normalized_tables = [
                        (box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height)
                        for box in table_regions or []
                    ]
                    if any(_significant_bbox_overlap(norm_bbox, table) for table in normalized_tables):
                        img_path.unlink(missing_ok=True)
                        print(f"    -> Ignored vector region {c_idx} overlapping an OCR table.")
                        continue
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
    # Vision models occasionally wrap one page fragment in a complete HTML
    # document shell. The shell is not content and must not split one table.
    markdown_text = re.sub(
        r"<!doctype\s+html(?:\s+[^>]*)?>|</?(?:html|body|head)\b[^>]*>|<meta\b[^>]*>",
        "",
        markdown_text,
        flags=re.IGNORECASE,
    )
    def sequence_rank(value: str) -> tuple[str, int] | None:
        """Parse a generic row-order key without assuming a document schema."""
        token = re.sub(r"<[^>]+>|[*_`]", "", value).strip().rstrip(".)-:")
        if re.fullmatch(r"\d+", token):
            return "number", int(token)
        if re.fullmatch(r"[A-Za-z]", token):
            return "letter", ord(token.casefold())
        if len(token) > 1 and re.fullmatch(r"[IVXLCDM]+", token, re.I):
            values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            total = previous = 0
            for character in reversed(token.upper()):
                current = values[character]
                total += -current if current < previous else current
                previous = max(previous, current)
            return "roman", total
        return None

    def trim_empty_trailing_columns(rows: list[str]) -> list[str]:
        """Drop only columns that are empty in every non-separator row."""
        parsed = [_split_markdown_table_row(row) for row in rows]
        data = [
            cells for cells in parsed
            if not all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell or "") for cell in cells)
        ]
        if not data:
            return rows
        width = max(len(cells) for cells in parsed)
        effective = width
        while effective > 1 and all(
            effective > len(cells) or not cells[effective - 1].strip()
            for cells in data
        ):
            effective -= 1
        if effective == width:
            return rows
        return [
            "| " + " | ".join((cells + [""] * effective)[:effective]) + " |"
            for cells in parsed
        ]

    def pad_rows(rows: list[str], width: int) -> list[str]:
        padded = []
        for row in rows:
            cells = _split_markdown_table_row(row)
            separator = all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell or "") for cell in cells)
            fill = ":---" if separator else ""
            cells.extend([fill] * max(0, width - len(cells)))
            padded.append("| " + " | ".join(cells[:width]) + " |")
        return padded

    def is_bridge_text(value: str) -> bool:
        residual = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
        residual = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", residual)
        residual = re.sub(
            r"(?im)^\s*(?:hình|ảnh|figure|nguồn\s*:)[^\n]*$", "", residual
        )
        return not residual.strip()

    # Join complete HTML tables separated only by page metadata/figures. Keep
    # the bridge after the combined table so image Markdown never sits inside
    # invalid <table> markup.
    html_table_re = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
    while True:
        matches = list(html_table_re.finditer(markdown_text))
        merged = False
        for left, right in zip(matches, matches[1:]):
            bridge = markdown_text[left.end():right.start()]
            if not is_bridge_text(bridge):
                continue
            first = left.group(0)
            second = right.group(0)
            combined = (
                first[: first.lower().rfind("</table>")]
                + second[second.find(">") + 1 : second.lower().rfind("</table>")]
                + "</table>"
                + bridge
            )
            markdown_text = markdown_text[:left.start()] + combined + markdown_text[right.end():]
            merged = True
            break
        if not merged:
            break

    lines = markdown_text.splitlines()
    blocks = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        is_table_start = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") > 1
        if is_table_start:
            has_header = False
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                has_header = next_stripped.startswith("|") and next_stripped.endswith("|") and all(c in " -:+|" for c in next_stripped)
            if has_header or blocks:
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

                if table_rows:
                    header_cells = [c.strip().casefold() for c in _split_markdown_table_row(table_rows[0])]
                    cleaned_rows = [table_rows[0]]
                    skip_next_if_sep = False
                    for r in table_rows[1:]:
                        cur_cells = [c.strip().casefold() for c in _split_markdown_table_row(r)]
                        if cur_cells == header_cells:
                            skip_next_if_sep = True
                            continue
                        if skip_next_if_sep and all(c in " -:+|" for c in "".join(cur_cells)):
                            skip_next_if_sep = False
                            continue
                        skip_next_if_sep = False
                        cleaned_rows.append(r)
                    table_rows = cleaned_rows

                blocks.append({
                    "type": "table",
                    "rows": table_rows,
                    "comments": table_comments,
                    "has_header": has_header,
                })
                continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            blocks.append({"type": "comment", "line": line})
        elif re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", stripped):
            blocks.append({"type": "comment", "line": line})
        elif re.match(r"^(?:Hình|Ảnh|Figure|Nguồn\s*:)", stripped, re.IGNORECASE):
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
                last["rows"] = trim_empty_trailing_columns(last["rows"])
                block["rows"] = trim_empty_trailing_columns(block["rows"])
                last_columns = len(_split_markdown_table_row(last["rows"][0]))
                current_columns = len(_split_markdown_table_row(block["rows"][0]))
                last_first = _split_markdown_table_row(last["rows"][0])[0].strip()
                current_first = _split_markdown_table_row(block["rows"][0])[0].strip()
                if (
                    last_columns > current_columns
                    and sequence_rank(last_first) is None
                    and sequence_rank(current_first) is not None
                ):
                    block["rows"] = pad_rows(block["rows"], last_columns)
                    current_columns = last_columns
                header_last = [c.strip().casefold() for c in _split_markdown_table_row(last["rows"][0])]
                header_curr = [c.strip().casefold() for c in _split_markdown_table_row(block["rows"][0])]
                same_header = block["has_header"] and header_last == header_curr
                continuation = not block["has_header"] and last_columns == current_columns

                is_pseudo_header = False
                if block["has_header"] and last_columns == current_columns and not same_header:
                    first_cell_curr = header_curr[0].strip() if header_curr else ""
                    first_cell_last_hdr = header_last[0].strip() if header_last else ""
                    last_data_row = _split_markdown_table_row(last["rows"][-1]) if last["rows"] else []
                    last_order_cell = last_data_row[0].strip() if last_data_row else ""
                    current_rank = sequence_rank(first_cell_curr)
                    header_rank = sequence_rank(first_cell_last_hdr)
                    previous_rank = sequence_rank(last_order_cell)

                    if current_rank is not None and header_rank is None:
                        is_pseudo_header = True
                    elif (
                        current_rank is not None and previous_rank is not None
                        and current_rank[0] == previous_rank[0]
                        and current_rank[1] == previous_rank[1] + 1
                    ):
                        is_pseudo_header = True

                if same_header or continuation or is_pseudo_header:
                    if same_header:
                        data_rows = block["rows"][2:]
                    elif is_pseudo_header:
                        data_rows = [block["rows"][0]] + block["rows"][2:]
                    else:
                        data_rows = block["rows"]
                    last["rows"].extend(data_rows)
                    last["comments"].extend(
                        item["line"] for item in merged_blocks[found_table_idx + 1:]
                        if item["type"] == "comment"
                    )
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
    import html

    def escape_cell(cell: str) -> str:
        # Preserve explicit line breaks while escaping OCR text that could form
        # unintended HTML tags or entities.
        marker = "__QWEN_OCR_BR__"
        protected = re.sub(r"<br\s*/?>", marker, cell, flags=re.IGNORECASE)
        return html.escape(protected, quote=False).replace(marker, "<br>")

    output = ["<table>"]
    for row_index, cells in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        output.append("  <tr>")
        output.extend(f"    <{tag}>{escape_cell(cell)}</{tag}>" for cell in cells)
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

    # Markdown images with a placeholder target, including paths such as
    # images/image_placeholder.png. Surplus placeholders are removed.
    markdown = re.sub(
        rf"!\[([^\]]*)\]\(\s*{IMAGE_PLACEHOLDER_TARGET_RE}\s*\)",
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
        rf"<img\b[^>]*\bsrc\s*=\s*(['\"]){IMAGE_PLACEHOLDER_TARGET_RE}\1[^>]*>",
        replace_html_image,
        markdown,
        flags=re.IGNORECASE,
    )

    # A bare placeholder is also valid model output. Convert it in place so
    # reading order is retained.
    markdown = re.sub(
        rf"(?<![\w/.-]){IMAGE_PLACEHOLDER_TARGET_RE}(?![\w/.-])",
        lambda _match: next_image("Hình ảnh"),
        markdown,
        flags=re.IGNORECASE,
    )

    remaining = [path for path in available if path not in used]
    return markdown, remaining


def remove_orphan_image_placeholders(markdown: str) -> str:
    """Remove unresolved image placeholders after all real assets were linked."""
    markdown = re.sub(
        rf"!\[[^\]]*\]\(\s*{IMAGE_PLACEHOLDER_TARGET_RE}\s*\)", "", markdown,
        flags=re.IGNORECASE,
    )
    markdown = re.sub(
        rf"<img\b[^>]*\bsrc\s*=\s*(['\"]){IMAGE_PLACEHOLDER_TARGET_RE}\1[^>]*>", "", markdown,
        flags=re.IGNORECASE,
    )
    markdown = re.sub(
        rf"(?<![\w/.-]){IMAGE_PLACEHOLDER_TARGET_RE}(?![\w/.-])", "", markdown,
        flags=re.IGNORECASE,
    )
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def normalize_escaped_image_links(markdown: str) -> str:
    """Repair image links whose opening parenthesis/path was escaped by Qwen."""
    def repair(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        image_path = match.group(2).replace(r"\_", "_")
        return f"![{alt_text}]({image_path})"

    # Restrict cleanup to image syntax so LaTeX, code and literal backslashes
    # elsewhere in the OCR remain unchanged.
    markdown = re.sub(
        r"!\[([^\]\n]*)\]\\\(([^)\n]*?)(?:\\)?\)",
        repair,
        markdown,
    )
    # Four leading spaces turn a standalone image into a Markdown code block.
    # Keep indentation elsewhere intact and only lift complete image-only lines.
    return re.sub(
        r"(?m)^[ \t]{4,}(!\[[^\]\n]*\]\([^)\n]+\))[ \t]*$",
        r"\1",
        markdown,
    )


def _insert_images_by_reading_order(
    markdown: str, image_paths: list[str], typed_blocks=None,
) -> str:
    """Place unreferenced content images near their layout position."""
    pending = [path for path in image_paths if f"]({path})" not in markdown]
    if not pending:
        return markdown
    parts = [part for part in re.split(r"\n{2,}", markdown.strip()) if part.strip()]
    if not parts:
        return "\n\n".join(f"![Hình ảnh]({path})" for path in pending)

    image_positions: list[float] = []
    if typed_blocks:
        blocks = list(typed_blocks)
        denominator = max(len(blocks), 1)
        def block_kind(block) -> str:
            return str(block.kind) if hasattr(block, "kind") else str(block[0])
        image_positions = [
            index / denominator for index, block in enumerate(blocks)
            if block_kind(block) == "image"
        ]
    insertions: list[tuple[int, str]] = []
    for path in pending:
        original_index = image_paths.index(path)
        fraction = image_positions[original_index] if original_index < len(image_positions) else 1.0
        target = min(len(parts), max(0, round(fraction * len(parts))))
        insertions.append((target, f"![Hình ảnh]({path})"))
    for target, tag in reversed(insertions):
        parts.insert(target, tag)
    return "\n\n".join(parts)


def apply_page_assets(
    markdown: str, page_number: int, image_paths: list[str],
    formulas: list[str] | None = None, typed_blocks=None,
) -> str:
    """Resolve placeholders and preserve all accepted content images."""
    if formulas is not None:
        iterator = iter(formulas)
        unused_formulas = list(formulas)

        def replace_formula(match):
            try:
                value = next(iterator)
                unused_formulas.pop(0)
                return value
            except StopIteration:
                return ""

        markdown = re.sub(r"formula_placeholder", replace_formula, markdown, flags=re.IGNORECASE)
        if unused_formulas:
            markdown = f"{markdown.rstrip()}\n\n" + "\n\n".join(unused_formulas)
    pending_images = [path for path in image_paths if f"]({path})" not in markdown]
    markdown, _unplaced = link_extracted_images(markdown, pending_images)
    markdown = _insert_images_by_reading_order(markdown, image_paths, typed_blocks)
    # Surplus placeholders have no valid asset and must never reach output.
    markdown = remove_orphan_image_placeholders(markdown)
    # Correct model-authored zero-based/stale generic page labels as well as
    # labels created by our own fallback path. Descriptive alt text is kept.
    markdown = re.sub(
        r"(!\[\s*Hình ảnh trang\s+)\d+(\s*\]\([^)]+\))",
        rf"\g<1>{page_number}\g<2>",
        markdown,
        flags=re.IGNORECASE,
    )
    return markdown


def cleanup_unreferenced_assets(markdown: str, image_paths: list[str], output_dir: str | Path) -> int:
    """Delete extracted local assets omitted from the final selected Markdown."""
    root = Path(output_dir).resolve()
    removed = 0
    for path in image_paths:
        if f"]({path})" in markdown or re.match(r"^(?:https?://|data:)", path, re.IGNORECASE):
            continue
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if target.is_file():
            target.unlink()
            removed += 1
    return removed


def deduplicate_exact_assets(
    markdown: str, image_paths: list[str], output_dir: str | Path,
) -> tuple[str, list[str], int]:
    """Collapse byte-identical images only; near-duplicates are kept."""
    root = Path(output_dir).resolve()
    digest_to_path: dict[str, str] = {}
    unique: list[str] = []
    removed = 0
    for path in image_paths:
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            unique.append(path)
            continue
        if not target.is_file():
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        canonical = digest_to_path.get(digest)
        if canonical is None:
            digest_to_path[digest] = path
            unique.append(path)
            continue
        escaped = re.escape(path)
        markdown = re.sub(
            rf"!\[[^\]]*\]\(\s*{escaped}\s*\)", "", markdown,
            flags=re.IGNORECASE,
        )
        markdown = re.sub(
            rf"<img\b[^>]*\bsrc\s*=\s*(['\"]){escaped}\1[^>]*>", "", markdown,
            flags=re.IGNORECASE,
        )
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        target.unlink()
        removed += 1
    return markdown, unique, removed


# Hàm dọn dẹp và chuẩn hóa văn bản Markdown nhận diện từ mô hình Vision
def clean_markdown(text: str) -> str:
    """
    Làm sạch Markdown đầu ra: loại bỏ rào chắn mã, lời thừa nhận diện của chatbot,
    chuẩn hóa thực thể khoảng trắng HTML, sửa công thức toán học bị nhận diện nhầm.
    """
    import re
    text = text.strip()
    text = re.sub(r"\\+hfill\b\s*", " ", text)
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\r?\n?", "", text)
    text = re.sub(r"\r?\n?```\s*$", "", text)
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
    from app.core.math_cleanup import normalize_answer_math
    return normalize_answer_math(text)


def post_process_markdown(text: str) -> str:
    """Apply final Markdown repairs shared by CLI and GUI."""
    text = re.sub(r"&(?:nbsp|amp);", " ", text)
    text = re.sub(r"\\+hfill\b\s*", " ", text)
    # Repair delimiter leakage at HTML cell boundaries before balancing each
    # cell. These patterns occur in variation/sign tables emitted by small VLMs.
    text = re.sub(
        r"(<td\b[^>]*>)\s*([+-])\$\s*(</td>)",
        r"\1$\2$\3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"</td>\s*\$\s*(?=<td\b)", "</td>", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(<td\b[^>]*>)\s*\$\s*(</td>)\s*\$?",
        r"\1\2",
        text,
        flags=re.IGNORECASE,
    )
    def balance_math(fragment: str) -> str:
        dollar_indices = [
            index for index, char in enumerate(fragment)
            if char == "$" and (index == 0 or fragment[index - 1] != "\\")
        ]
        if len(dollar_indices) % 2 == 0:
            return fragment
        tail = fragment[dollar_indices[-1] + 1:]
        # Do not turn a currency amount such as "$5" into mathematics.
        if not re.search(r"[=+*/^_{}\\<>]|[A-Za-zÀ-ỹĐđ]\s*\d|\d\s*[A-Za-zÀ-ỹĐđ]", tail):
            return fragment
        stripped = fragment.rstrip()
        punctuation = re.search(r"([.,;:!?])$", stripped)
        return stripped[:-1] + "$" + stripped[-1] if punctuation else stripped + "$"

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
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = _split_markdown_table_row(line)
            balanced_cells = [balance_math(cell) for cell in cells]
            line = "| " + " | ".join(balanced_cells) + " |"
        elif re.search(r"<(?:td|th)\b", line, re.IGNORECASE):
            line = re.sub(
                r"(<(?:td|th)\b[^>]*>)(.*?)(</(?:td|th)>)",
                lambda match: match.group(1) + balance_math(match.group(2)) + match.group(3),
                line,
                flags=re.IGNORECASE,
            )
        else:
            line = balance_math(line)
        processed_lines.append(line)

    result = "\n".join(processed_lines)
    result = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: match.group(1)
        if len(match.group(1).split()) >= 6
        and len(re.findall(r"[=+*/^_{}\\]", match.group(1))) <= 2
        else match.group(0),
        result,
    )
    from app.core.math_cleanup import normalize_answer_math
    return normalize_answer_math(result)


from html.parser import HTMLParser

class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.has_complex_span = False
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_map = dict(attrs)
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []
            colspan = attrs_map.get("colspan")
            rowspan = attrs_map.get("rowspan")
            if (colspan and colspan != "1") or (rowspan and rowspan != "1"):
                self.has_complex_span = True
        elif tag == "br" and self.in_cell:
            self.current_cell.append("<br>")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self.in_cell = False
            cell_text = "".join(self.current_cell).strip()
            cell_text = re.sub(r"\r?\n+", "<br>", cell_text)
            self.current_row.append(cell_text)
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def convert_simple_html_tables_to_markdown(markdown: str) -> tuple[str, int]:
    """Tự động chuyển đổi các bảng HTML đơn giản thành bảng Markdown pipe table (|)."""
    table_re = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
    converted_count = 0

    def replace_table(match: re.Match[str]) -> str:
        nonlocal converted_count
        html_code = match.group(0)
        # Nếu có bảng lồng nhau, giữ nguyên HTML
        if len(re.findall(r"<table\b", html_code, re.IGNORECASE)) > 1:
            return html_code

        parser = _SimpleHTMLTableParser()
        try:
            parser.feed(html_code)
        except Exception:
            return html_code

        if parser.has_complex_span or not parser.rows or len(parser.rows) < 2:
            return html_code

        col_counts = [len(r) for r in parser.rows]
        if not col_counts:
            return html_code

        num_cols = max(col_counts)
        if num_cols < 2:
            return html_code
        if len(parser.rows[0]) != num_cols:
            # A data row wider than the declared header is structurally
            # ambiguous; leave it as repaired HTML rather than invent columns.
            return html_code

        # Page continuations often omit trailing empty cells (for example an
        # unused "Ghi chú" column). Preserve the common schema by padding only
        # the missing trailing cells instead of exposing the HTML as code.
        parser.rows = [row + [""] * (num_cols - len(row)) for row in parser.rows]


        md_lines = []
        header_row = parser.rows[0]
        header_normalized = [c.strip().casefold() for c in header_row]
        md_lines.append("| " + " | ".join(c.replace("|", "\\|") for c in header_row) + " |")
        md_lines.append("| " + " | ".join(":---" for _ in range(num_cols)) + " |")

        for data_row in parser.rows[1:]:
            row_normalized = [c.strip().casefold() for c in data_row]
            if row_normalized == header_normalized:
                continue
            md_lines.append("| " + " | ".join(c.replace("|", "\\|") for c in data_row) + " |")

        converted_count += 1
        return "\n\n" + "\n".join(md_lines) + "\n\n"


    result = table_re.sub(replace_table, markdown)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result, converted_count


def finalize_markdown(
    markdown: str, *, return_report: bool = False, spell_correct: bool = False
):
    """Canonical finalization used by every frontend, optionally with a report."""
    from app.core.markdown_normalizer import MarkdownNormalizationStats, normalize_structure

    stats = MarkdownNormalizationStats()
    markdown = normalize_escaped_image_links(markdown)
    markdown = remove_orphan_image_placeholders(markdown)
    markdown = normalize_structure(markdown, stats)
    table_count_before = len(re.findall(r"<table\b", markdown, re.IGNORECASE))
    markdown = repair_markdown_tables(markdown)
    tables_repaired = max(
        0, len(re.findall(r"<table\b", markdown, re.IGNORECASE)) - table_count_before
    )
    markdown = merge_markdown_tables(markdown)

    # Tự động chuyển các bảng HTML đơn giản sang bảng Markdown pipe table chuẩn (|)
    markdown, tables_converted_to_md = convert_simple_html_tables_to_markdown(markdown)
    # A document can cross a page boundary as Markdown on one page and HTML on
    # the next.  Those tables only share a representation after conversion, so
    # run the merger again to join the newly normalized continuation.
    if tables_converted_to_md:
        markdown = merge_markdown_tables(markdown)

    before_math = markdown
    markdown = post_process_markdown(markdown)
    markdown = remove_orphan_image_placeholders(markdown)
    math_normalized = int(markdown != before_math)
    from app.core.vietnamese_spell_corrector import (
        correct_vietnamese_spelling, suggest_vietnamese_spelling,
    )
    spelling_warnings = suggest_vietnamese_spelling(markdown)
    spell_fixed_count = 0
    if spell_correct:
        markdown, spell_fixed_count = correct_vietnamese_spelling(markdown)
    report = {
        "headings": stats.headings,
        "headings_split": stats.headings_split,
        "paragraph_lines_joined": stats.paragraph_lines_joined,
        "list_lines_joined": stats.list_lines_joined,
        "tables_repaired": tables_repaired,
        "tables_converted_to_md": tables_converted_to_md,
        "html_tags_closed": stats.html_tags_closed,
        "math_normalized": math_normalized,
        "image_paths": stats.image_paths,
        "duplicate_images": stats.duplicate_images,
        "page_artifacts_removed": stats.page_artifacts_removed,
        "repetition_lines_removed": stats.repetition_lines_removed,
        "spell_fixed": spell_fixed_count,
        "spell_warnings": len(spelling_warnings),
        "spelling_warnings": spelling_warnings,
    }
    return (markdown, report) if return_report else markdown


def format_finalization_report(report: dict) -> list[str]:
    """Return concise Vietnamese log lines for Markdown finalization."""
    labels = {
        "headings": "Heading đã chuẩn hóa",
        "headings_split": "Heading dính dòng đã tách",
        "paragraph_lines_joined": "Dòng văn bản đã nối",
        "list_lines_joined": "Dòng danh sách đã nối",
        "tables_repaired": "Bảng lỗi đã chuyển sang HTML",
        "tables_converted_to_md": "Bảng HTML đơn giản đã chuyển sang Markdown",
        "html_tags_closed": "Thẻ HTML đã đóng bổ sung",
        "math_normalized": "Lượt chuẩn hóa công thức/LaTeX",
        "image_paths": "Đường dẫn ảnh đã chuẩn hóa",
        "duplicate_images": "Ảnh trùng đã loại bỏ",
        "page_artifacts_removed": "Header/footer/số trang đã loại bỏ",
        "repetition_lines_removed": "Dòng lặp hallucination đã loại bỏ",
        "spell_fixed": "Từ tiếng Việt đã sửa theo ngữ cảnh",
        "spell_warnings": "Từ nghi ngờ chính tả (chỉ cảnh báo)",
    }
    lines = [
        f"{labels[key]}: {value}" for key, value in report.items()
        if key in labels and value
    ]

    return lines or ["Không phát hiện lỗi định dạng cần sửa"]


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
    max_retries: int = 3,
) -> str:
    """Run one canonical Qwen OCR pass; retries are owned by the quality gate."""
    def consume(chunks) -> str:
        collected: list[str] = []
        try:
            for chunk in chunks:
                if before_request is not None:
                    before_request()
                collected.append(chunk.response)
        except BaseException:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
            raise
        return "".join(collected)

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
            max_retries=max_retries, log_func=log_func,
        )
        parts.append(clean_markdown(consume(chunks)))

    return "\n\n".join(part for part in parts if part.strip())


def ocr_coordinate_blocks(
    client, model: str, page_image: Path, typed_blocks, image_paths: list[str],
    output_dir: Path, *, page_number: int = 1, log_func=print, before_request=None,
    mapping_sink: list | None = None,
) -> str | None:
    """OCR the full page once and place extracted images by layout order."""
    from app.core.block_assembler import DocumentBlock

    if not isinstance(typed_blocks, list):
        return None
    blocks = [DocumentBlock(kind, tuple(bbox)) for kind, bbox in typed_blocks]
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
    return apply_page_assets(markdown, page_number, image_paths, typed_blocks=blocks)


def retain_extracted_image_blocks(
    typed_blocks: list[tuple[str, BoundingBox]], accepted_images: list[BoundingBox]
) -> list[tuple[str, BoundingBox]]:
    """Keep image blocks only when a corresponding crop survived filtering.

    OCR still receives the complete page. This prevents a broad layout image
    box (for example a stamp overlapping a printed name) from suppressing text
    after that graphical crop has intentionally been rejected.
    """
    accepted = set(accepted_images)
    return [
        (kind, bbox) for kind, bbox in typed_blocks
        if kind != "image" or bbox in accepted
    ]


def _significant_bbox_overlap(first: BoundingBox, second: BoundingBox) -> bool:
    """Return true when an image substantially duplicates a detected table region."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / first_area >= 0.35 or intersection / second_area >= 0.65


def normalized_document_stem(pdf_path: Path) -> str:
    """Return the source stem in composed Unicode without ASCII slugging."""
    return unicodedata.normalize("NFC", pdf_path.stem)


def output_markdown_path(output_dir: Path, pdf_path: Path) -> Path:
    """Build an output filename while preserving Vietnamese Unicode."""
    return output_dir / f"{normalized_document_stem(pdf_path)}.md"


# Hàm chính xử lý OCR cho một tệp PDF đơn lẻ
def process_single_pdf(
    pdf_path: Path, output_dir: Path, client: Client, model_name: str,
    workers: int = 1, render_dpi: int = 300,
):
    """
    Tiến hành lập trình tự render và nhận diện OCR toàn bộ tệp PDF:
    - Khởi tạo thư mục và quét số trang.
    - Chạy phân tích bố cục PaddleOCR để phát hiện bảng/cột.
    - Xử lý nhận diện và ghép nối nội dung.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if render_dpi < 72:
        raise ValueError("render_dpi must be at least 72")
    workers = normalize_worker_count(workers, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_markdown_path(output_dir, pdf_path)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"\nProcessing: {pdf_path.name} -> {output_path.name}")
    document_started_at = perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        total_pages = pdf_page_count(pdf_path)
        print(f"Pipelining render and OCR for {total_pages} pages (workers={workers})...")
    print(f"\nProcessing: {pdf_path.name} -> {output_path.name}")
    document_started_at = perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        total_pages = pdf_page_count(pdf_path)
        print(f"Pipelining render and OCR for {total_pages} pages (workers={workers})...")
        from app.core.formula_ocr import formula_ocr_status
        _, formula_status = formula_ocr_status()
        print(f"Formula OCR status: {formula_status}.")

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
            qwen_model = hybrid_model if is_hybrid else resolve_qwen_model(model_name)
            page_block_spans = []
            mapped_markdown = None

            print(f"OCR'ing page {page_num}/{total_pages}: {img_path.name}...")
            started_at = perf_counter()

            # Chuẩn hóa hướng trang trước mọi phép phân tích để Paddle Layout,
            # Qwen Vision, formula OCR và bước review dùng cùng một hệ tọa độ.
            page_rotation = 0
            try:
                page_rotation = orient_page_file(img_path)
                if page_rotation:
                    print(f"    -> Auto-rotated page {page_num} by {page_rotation} degrees.")
            except Exception as orientation_error:
                print(
                    f"    [Warning] Auto-rotate failed on page {page_num}; "
                    f"using rendered orientation: {orientation_error}"
                )

            page_md = ""
            error = None
            qwen_images = [img_path]
            has_table = False
            segments = None
            ordered_blocks = []
            page_analysis = None
            render_seconds = render_timings.get(img_path, 0.0)
            paddle_seconds = 0.0
            qwen_seconds = 0.0
            qwen_first_seconds = 0.0
            retry_seconds = 0.0
            formula_seconds = 0.0

            # Phân tích bố cục bằng PaddleOCR nếu bộ phát hiện được nạp thành công
            if layout_detector is not None:
                try:
                    paddle_started_at = perf_counter()
                    with layout_lock:
                        page_analysis = layout_detector.analyse_page(img_path)
                    segments = page_analysis.segments
                    has_table = bool(page_analysis.tables)
                    ordered_blocks = page_analysis.blocks
                    paddle_seconds = perf_counter() - paddle_started_at
                    if segments and len(segments) >= 2:
                        from app.core.layout_detector import crop_segments
                        qwen_images = crop_segments(img_path, segments, temp_dir_path / "segments")
                        print(f"    -> Layout detected {len(qwen_images)} segments on page {page_num}.")
                    if has_table:
                        # Cắt nhỏ bảng sẽ làm hỏng ngữ cảnh hàng/cột, dùng ảnh gốc độ phân giải cao
                        table_page = render_table_page(
                            pdf_path, idx, temp_dir_path / "table_pages"
                        )
                        if page_rotation:
                            with Image.open(table_page) as source:
                                oriented_table = source.convert("RGB").rotate(
                                    page_rotation, expand=True
                                )
                            oriented_table.save(table_page, format="PNG")
                            oriented_table.close()
                        qwen_images = [table_page]
                        print(f"    -> Layout detected table on page {page_num}; re-rendered at {TABLE_RENDER_DPI} DPI.")
                except Exception as layout_error:
                    print(f"    [Warning] Layout detection failed on page {page_num}; using full page: {layout_error}")
                    qwen_images, has_table = [img_path], False
                    segments = None
                    ordered_blocks = []

            # 1. Trích xuất hình ảnh vật lý từ trang PDF
            extracted_img_paths = []
            try:
                if TEXT_ONLY_OUTPUT:
                    if page_analysis is not None:
                        page_analysis.images.clear()
                    raise StopIteration
                output_img_dir = output_dir / "images"
                extracted_img_paths = extract_images_from_page(
                    pdf_path, idx, output_img_dir, normalized_document_stem(pdf_path),
                    layout_detector=layout_detector,
                    page_image_path=img_path,
                    segments=segments,
                    detected_images=page_analysis.images if page_analysis else [],
                    table_regions=page_analysis.tables if page_analysis else [],
                    graphic_classifier=lambda crop: classify_graphic_crop(
                        client, qwen_model, crop,
                        log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                    ),
                )
            except StopIteration:
                pass
            except Exception as img_err:
                print(f"    -> [Warning] Failed to extract images: {img_err}")

            if page_analysis is not None:
                ordered_blocks = retain_extracted_image_blocks(
                    ordered_blocks, page_analysis.images
                )

            # 1.5. Trích xuất công thức toán học từ trang PDF bằng LaTeX-OCR
            formulas_latex = []
            if layout_detector is not None and ENABLE_LAYOUT_DETECTION:
                formula_started_at = perf_counter()
                try:
                    detected_formulas = page_analysis.formulas if page_analysis else []
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
                finally:
                    formula_seconds = perf_counter() - formula_started_at

            # 2. Thực hiện OCR và làm sạch kết quả bằng Qwen
            qwen_model = hybrid_model if is_hybrid else resolve_qwen_model(model_name)
            try:
                qwen_started_at = perf_counter()
                assets_applied = False
                formula_instruction = (
                    "\n\nLƯU Ý CÔNG THỨC TOÁN HỌC: Hãy thay thế mọi công thức hoặc "
                    "biểu thức toán học phức tạp bằng nhãn chính xác: formula_placeholder."
                    if formulas_latex else ""
                )
                page_md = ocr_coordinate_blocks(
                    client, qwen_model, img_path, ordered_blocks, extracted_img_paths,
                    temp_dir_path / "layout_blocks" / f"page_{page_num}",
                    page_number=page_num,
                    log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                    mapping_sink=page_block_spans,
                )
                if page_md is not None:
                    mapped_markdown = page_md
                    assets_applied = True
                    print(f"    -> OCRed page {page_num} once and placed images in reading order.")
                else:
                    page_md = ocr_qwen_images(
                        client, qwen_model, qwen_images, has_table=has_table,
                        extra_instruction=formula_instruction,
                        log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                    )
                qwen_seconds = perf_counter() - qwen_started_at
                qwen_first_seconds = qwen_seconds
            except Exception as ollama_err:
                qwen_seconds = perf_counter() - qwen_started_at
                error = f"Qwen OCR failed: {ollama_err}"

            # 2.5. Thay thế placeholder công thức và hình ảnh theo thứ tự đọc
            if extracted_img_paths:
                print(f"    -> Extracted {len(extracted_img_paths)} images from PDF page {page_num}.")
            if page_md:
                page_md = apply_page_assets(
                    page_md, page_num,
                    [] if assets_applied else extracted_img_paths,
                    formulas_latex,
                )

            # 3. Quality gate: retry only this page once when under quality threshold.
            if not error:
                from app.core.quality_gate import choose_best_page, evaluate_page
                report = evaluate_page(page_md, output_dir, check_tables=has_table)
                errors_str = f"Errors: {', '.join(report.errors)}" if report.errors else "No errors"
                warnings_str = f", Warnings: {', '.join(report.warnings)}" if report.warnings else ""
                print(f"    -> [Quality Gate Page {page_num}] Score: {report.score}/100.0 ({errors_str}{warnings_str})")
                if report.warnings:
                    print(f"    -> [Warning] Page {page_num}: {', '.join(report.warnings)}")
                if report.should_retry:
                    initial_md, initial_report = page_md, report
                    print(f"    -> Quality retry page {page_num} (Score: {report.score}/100 < threshold or fatal): {', '.join(report.errors)}")
                    retry_started_at = perf_counter()
                    try:
                        retry_md = ocr_qwen_images(
                            client, qwen_model, [img_path], has_table=has_table,
                            extra_instruction="\n\n" + quality_retry_instruction(initial_md, ", ".join(report.errors)),
                            log_func=lambda message: print(f"    -> [Page {page_num}] {message}"),
                        )
                        retry_md = apply_page_assets(retry_md, page_num, extracted_img_paths, formulas_latex)
                        second_report = evaluate_page(retry_md, output_dir, check_tables=has_table)
                        second_errors_str = f"Errors: {', '.join(second_report.errors)}" if second_report.errors else "No errors"
                        second_warnings_str = f", Warnings: {', '.join(second_report.warnings)}" if second_report.warnings else ""
                        print(f"    -> [Quality Gate Retry Page {page_num}] Score: {second_report.score}/100.0 ({second_errors_str}{second_warnings_str})")
                        if second_report.warnings:
                            print(f"    -> [Warning] Retry page {page_num}: {', '.join(second_report.warnings)}")
                        page_md, report = choose_best_page(
                            initial_md, initial_report, retry_md, second_report
                        )
                        if page_md == retry_md:
                            page_block_spans = []

                    except Exception as retry_error:
                        page_md, report = initial_md, initial_report
                        print(f"    -> [Warning] Quality retry failed on page {page_num}; keeping original result: {retry_error}")
                    finally:
                        retry_seconds = perf_counter() - retry_started_at
                    if not report.passed:
                        print(
                            f"    -> [Warning] Page {page_num} still failed quality gate "
                            f"({', '.join(report.errors)}); using best result and continuing."
                        )

            page_md, extracted_img_paths, duplicate_assets = deduplicate_exact_assets(
                page_md, extracted_img_paths, output_dir,
            )
            if duplicate_assets:
                print(f"    -> Collapsed {duplicate_assets} byte-identical image(s) on page {page_num}.")
            removed_assets = cleanup_unreferenced_assets(page_md, extracted_img_paths, output_dir)
            if removed_assets:
                print(f"    -> Removed {removed_assets} unreferenced extracted image(s) from page {page_num}.")

            qwen_seconds = perf_counter() - qwen_started_at if 'qwen_started_at' in locals() else qwen_seconds
            elapsed = perf_counter() - started_at
            print(
                f"    Benchmark: Render {render_seconds:.1f}s | Layout {paddle_seconds:.1f}s | "
                f"Qwen first {qwen_first_seconds:.1f}s | Retry {retry_seconds:.1f}s | "
                f"Formula OCR {formula_seconds:.1f}s"
            )
            print(f"Page {page_num} done in {elapsed:.1f}s.")
            review_regions[page_num] = [bbox for kind, bbox in ordered_blocks if kind != "image"]
            review_graphics[page_num] = [bbox for kind, bbox in ordered_blocks if kind == "image"]
            if not error:
                review_quality_scores[page_num] = report.score
            if page_block_spans and mapped_markdown is not None and page_md != mapped_markdown:
                from app.core.block_assembler import rebase_block_spans
                page_block_spans = rebase_block_spans(mapped_markdown, page_md, page_block_spans)
            review_block_spans[page_num] = page_block_spans
            return page_num, page_md, error

        # Gửi tác vụ OCR ngay khi từng trang được render xong
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            page_image_paths: dict[int, Path] = {}
            review_regions: dict[int, list[BoundingBox]] = {}
            review_graphics: dict[int, list[BoundingBox]] = {}
            review_quality_scores: dict[int, float] = {}
            review_block_spans: dict[int, list] = {}
            for index, image_path in enumerate(iter_render_pdf_to_images(
                pdf_path, temp_dir_path, dpi=render_dpi, render_timings=render_timings,
            )):
                page_image_paths[index + 1] = image_path
                futures.append(executor.submit(process_page_worker, (index, image_path)))
            results = [future.result() for future in futures]

        # Sắp xếp lại theo đúng thứ tự số trang ban đầu
        results.sort(key=lambda x: x[0])
        from app.core.image_grounded_review import review_suspicious_lines
        reviewed_results = []
        for page_num, page_md, error in results:
            if page_md and not error:
                try:
                    page_md = review_suspicious_lines(
                        client, resolve_qwen_model(model_name), page_image_paths[page_num], page_md,
                        temp_dir_path / "review_crops" / f"page_{page_num}",
                        log_func=lambda message, number=page_num: print(f"    -> [Page {number}] {message}"),
                        regions=review_regions.get(page_num),
                        graphic_regions=review_graphics.get(page_num),
                        block_spans=review_block_spans.get(page_num) or None,
                        review_document_footer=(
                            page_num == total_pages and bool(review_graphics.get(page_num))
                        ),
                        quality_score=review_quality_scores.get(page_num),
                    )
                except Exception as review_error:
                    print(f"    -> [Warning] Image-grounded review failed on page {page_num}; kept OCR result: {review_error}")
            reviewed_results.append((page_num, page_md, error))
        results = reviewed_results
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
        finalization_report = {"spelling_warnings": []}
        try:
            final_md, finalization_report = finalize_markdown(final_md, return_report=True)
            print("    -> Hậu xử lý Markdown:")
            for report_line in format_finalization_report(finalization_report):
                print(f"       - {report_line}")
        except Exception as merge_err:
            print(f"    -> [Warning] Failed to finalize Markdown: {merge_err}")
        write_started_at = perf_counter()
        temp_output_path.write_text(final_md, encoding="utf-8")
        os.replace(temp_output_path, output_path)
        write_seconds = perf_counter() - write_started_at
        print(f"Saved OCR to {output_path}")
        print(f"Write benchmark: {write_seconds:.3f}s")
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
    parser.add_argument("--dpi", type=int, choices=(200, 250, 300), default=300, help="PDF render DPI (default: 300)")
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
    client = Client(host="http://localhost:11434", timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)

    total_start = perf_counter()
    for pdf_file in pdf_files:
        process_single_pdf(
            pdf_file, ocr_dir, client, args.model,
            workers=args.workers, render_dpi=args.dpi,
        )

    total_elapsed = perf_counter() - total_start
    print(f"\nBatch OCR processing completed in {total_elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()

