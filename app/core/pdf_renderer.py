"""
Module: pdf_renderer.py
Nhiệm vụ: Chuyển đổi các trang của tệp tài liệu PDF thành hình ảnh PNG (page_00001.png, page_00002.png, ...) sử dụng PyMuPDF.
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
import pymupdf


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
    
    saved_images = []
    doc = pymupdf.open(pdf_path)
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            # Render trang thành pixmap với DPI chỉ định
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            
            # Tên file ảnh đầu ra dạng page_00001.png
            image_name = f"page_{index + 1:05d}.png"
            image_path = output_dir / image_name
            
            pix.save(str(image_path))
            saved_images.append(image_path)
    finally:
        doc.close()
        
    return saved_images

