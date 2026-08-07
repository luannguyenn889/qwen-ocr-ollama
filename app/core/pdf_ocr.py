"""
Module: pdf_ocr
Nhiệm vụ: Phối hợp quá trình nhận diện toàn bộ tài liệu PDF (OCR PDF).
Quy trình: 
  1. Chuyển đổi từng trang PDF thành hình ảnh tạm thời (PNG).
  2. Gửi các hình ảnh qua Ollama Engine để nhận diện văn bản.
  3. Ghép kết quả nhận diện của từng trang vào một file Markdown (.md) duy nhất.
"""

import os
import tempfile
from pathlib import Path
from time import perf_counter

from app.core.ollama_engine import (
    OllamaQwenEngine,
)

from app.core.pdf_rerender import (
    render_pdf,
)


def ocr_pdf(
    engine: OllamaQwenEngine,
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
):
    """
    Hàm chính xử lý OCR cho toàn bộ tệp PDF:
    - Nhận vào đối tượng engine, đường dẫn tệp PDF và thư mục lưu kết quả.
    - Lưu kết quả ra file markdown (.md) trùng tên với file PDF tại output_dir.
    """
    pdf_path = Path(
        pdf_path
    ).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file PDF đầu vào: {pdf_path}"
        )

    output_dir = Path(
        output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        output_dir
        / f"{pdf_path.stem}.md"
    )

    temp_path = (
        output_dir
        / f"{pdf_path.stem}.md.tmp"
    )

    print(f"[1/3] PDF đầu vào: {pdf_path}", flush=True)
    print(f"      Kích thước: {pdf_path.stat().st_size:,} bytes", flush=True)
    print("[2/3] Đang kiểm tra kết nối Ollama...", flush=True)
    engine.check_connection()
    print(f"[3/3] Đang render và OCR PDF (DPI={dpi})...", flush=True)
    document_started_at = perf_counter()

    with tempfile.TemporaryDirectory() as temp_images:

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for page_number, image_path in render_pdf(
                pdf_path,
                temp_images,
                dpi=dpi,
            ):

                print(
                    f"[Trang {page_number}] Đã render ảnh, đang OCR bằng Ollama...",
                    flush=True,
                )
                page_started_at = perf_counter()

                markdown = engine.ocr_image(
                    image_path
                )

                page_elapsed = perf_counter() - page_started_at
                print(
                    f"[Trang {page_number}] Hoàn tất trong {page_elapsed:.1f} giây.",
                    flush=True,
                )

                output_file.write(
                    f"\n\n<!-- Trang {page_number} -->\n\n"
                )

                output_file.write(
                    markdown
                )

                output_file.write(
                    "\n"
                )

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
