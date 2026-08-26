"""
Module: pdf_ocr.py
Nhiệm vụ: Phối hợp quá trình nhận diện toàn bộ tài liệu PDF (OCR PDF).
Quy trình:
  1. Dùng text layer có sẵn của PDF khi đủ tin cậy.
  2. Chỉ render/OCR các trang scan hoặc không có text layer.
  3. Ghép kết quả vào một file Markdown (.md) duy nhất.
"""

import os
import tempfile
from pathlib import Path
from time import perf_counter

# pyrefly: ignore [missing-import]
import pymupdf

from app.core.ollama_engine import (
    OllamaQwenEngine,
)

from app.core.pdf_renderer import (
    render_pdf,
)
from app.core.pdf_text_layer import extract_native_text
from app.core.batch_ocr import finalize_markdown, output_markdown_path


# Hàm chính điều phối quá trình xử lý OCR cho toàn bộ file PDF
def ocr_pdf(
    engine: OllamaQwenEngine,
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
):
    """
    Hàm chính xử lý OCR cho toàn bộ tệp PDF:
    - Nhận vào đối tượng engine, đường dẫn tệp PDF và thư mục lưu kết quả.
    - Lưu kết quả ra file markdown (.md) trùng tên với file PDF tại output_dir.
    """
    # Chuyển đổi và chuẩn hóa đường dẫn đầu vào
    pdf_path = Path(
        pdf_path
    ).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file PDF đầu vào: {pdf_path}"
        )

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = Path(
        output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Xác định đường dẫn file đích và file tạm
    final_path = output_markdown_path(output_dir, pdf_path)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    print(f"[1/3] PDF đầu vào: {pdf_path}", flush=True)
    print(f"      Kích thước: {pdf_path.stat().st_size:,} bytes", flush=True)

    # Phân loại trang: trang nào có text sẵn, trang nào cần OCR
    native_pages: dict[int, str] = {}
    ocr_page_numbers: set[int] = set()
    document = pymupdf.open(pdf_path)
    try:
        for index in range(document.page_count):
            page_number = index + 1
            native_text = extract_native_text(document.load_page(index))
            if native_text.is_usable:
                native_pages[page_number] = native_text.markdown
            else:
                ocr_page_numbers.add(page_number)
    finally:
        document.close()

    print(
        f"[2/3] Text layer: {len(native_pages)} trang; OCR fallback: {len(ocr_page_numbers)} trang.",
        flush=True,
    )

    # Kiểm tra kết nối engine trước khi thực hiện OCR các trang cần thiết
    if ocr_page_numbers:
        print("      Đang kiểm tra kết nối Ollama...", flush=True)
        engine.check_connection()

    print(f"[3/3] Đang xuất Markdown (OCR fallback DPI={dpi})...", flush=True)
    document_started_at = perf_counter()

    # Sử dụng thư mục tạm để lưu ảnh render trước khi OCR
    with tempfile.TemporaryDirectory() as temp_images:

        # Render các trang cần OCR thành ảnh
        rendered_pages = dict(render_pdf(
            pdf_path, temp_images, dpi=dpi, page_numbers=ocr_page_numbers,
        ))
        document_parts: list[str] = []

        # Duyệt qua từng trang để ghép nội dung từ text layer hoặc OCR
        for page_number in range(1, len(native_pages) + len(ocr_page_numbers) + 1):
            if page_number in native_pages:
                markdown = native_pages[page_number]
                source = "native-text"
                print(f"[Trang {page_number}] Dùng text layer PDF; bỏ qua OCR.", flush=True)
            else:
                image_path = rendered_pages[page_number]
                print(f"[Trang {page_number}] Đã render ảnh, đang OCR bằng Ollama...", flush=True)
                page_started_at = perf_counter()
                markdown = engine.ocr_image(image_path)
                page_elapsed = perf_counter() - page_started_at
                source = "ocr"
                print(f"[Trang {page_number}] Hoàn tất OCR trong {page_elapsed:.1f} giây.", flush=True)
            document_parts.append(
                f"<!-- Trang {page_number}; nguồn: {source} -->\n\n{markdown}"
            )

        finalized = finalize_markdown("\n\n".join(document_parts)) + "\n"
        temp_path.write_text(finalized, encoding="utf-8")

    # Hoàn tất: đổi tên file tạm thành file chính thức
    os.replace(
        temp_path,
        final_path,
    )

    total_elapsed = perf_counter() - document_started_at
    print(
        f"Hoàn tất OCR PDF trong {total_elapsed:.1f} giây. "
        f"Đã lưu: {final_path}",
        flush=True,
    )

    return final_path
