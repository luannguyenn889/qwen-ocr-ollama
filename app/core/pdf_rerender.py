"""
Module: pdf_rerender
Nhiệm vụ: Chuyển đổi các trang của tệp tài liệu PDF thành hình ảnh độ phân giải cao (PNG) bằng thư viện PyMuPDF (fitz), phục vụ làm đầu vào cho mô hình OCR.
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
import pymupdf


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
):

    pdf_path = Path(pdf_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = pymupdf.open(
        pdf_path
    )

    try:

        for index in range(
            document.page_count
        ):

            page = document.load_page(
                index
            )

            pix = page.get_pixmap(
                dpi=dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )

            image_path = (
                output_dir
                / f"page_{index + 1:05d}.png"
            )

            pix.save(
                str(image_path)
            )

            yield (
                index + 1,
                image_path,
            )

    finally:
        document.close()