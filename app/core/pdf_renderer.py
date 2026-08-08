"""
Module: pdf_renderer
Nhiệm vụ: Chuyển đổi các trang của tệp tài liệu PDF thành hình ảnh PNG (page_00001.png, page_00002.png, ...) sử dụng PyMuPDF.
"""

from pathlib import Path
# pyrefly: ignore [missing-import]
import pymupdf

def render_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    # dpi: int = 200,
    dpi: int = 300,
) -> list[Path]:
    """
    Chuyển đổi từng trang của file PDF thành ảnh PNG.
    Lưu dưới định dạng: page_00001.png, page_00002.png, ...
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
