"""
Module: pdf_renderer.py
Nhiệm vụ: Chuyển đổi các trang của tệp tài liệu PDF thành hình ảnh PNG (page_00001.png, page_00002.png, ...) sử dụng PyMuPDF.
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
import pymupdf


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    page_numbers: set[int] | None = None,
):

    """Yield ``(one_based_page_number, image_path)`` for selected PDF pages."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    try:
        for index in range(document.page_count):
            page_number = index + 1
            if page_numbers is not None and page_number not in page_numbers:
                continue
            pix = document.load_page(index).get_pixmap(
                dpi=dpi, colorspace=pymupdf.csRGB, alpha=False,
            )
            image_path = output_dir / f"page_{page_number:05d}.png"
            pix.save(str(image_path))
            yield page_number, image_path
    finally:
        document.close()


# Hàm render toàn bộ PDF thành mảng đường dẫn file ảnh PNG
def render_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
) -> list[Path]:
    """
    Chuyển đổi từng trang của file PDF thành ảnh ảnh PNG và lưu xuống đĩa.
    Lưu dưới định dạng tên file: page_00001.png, page_00002.png, ...
    
    Tham số:
      - pdf_path: Đường dẫn vật lý của tệp PDF.
      - output_dir: Thư mục chứa các tệp ảnh kết quả.
      - dpi: Độ phân giải khi kết xuất hình ảnh (mặc định là 300 DPI).
      
    Trả về: Danh sách các đường dẫn Path tới các tệp ảnh PNG đã tạo.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return [image_path for _, image_path in render_pdf(pdf_path, output_dir, dpi=dpi)]
