"""
Module: step6_render.py
Nhiệm vụ: Kết xuất tài liệu PDF thành các trang hình ảnh PNG đơn lẻ với tên tệp đơn giản.
"""

import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF

# Hàm render trang PDF thành các tệp hình ảnh
def render_pdf_to_simple_names(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Kết xuất từng trang trong tệp PDF thành ảnh PNG dạng page_1.png, page_2.png..."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_images = []
    
    doc = fitz.open(pdf_path)
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            # Render trang thành pixmap màu RGB không kênh alpha
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            
            # Tên đơn giản: page_1.png, page_2.png, ...
            image_name = f"page_{index + 1}.png"
            image_path = output_dir / image_name
            
            pix.save(str(image_path))
            saved_images.append(image_path)
            print(f"Rendered: {image_path.name}")
    finally:
        doc.close()
        
    return saved_images

# Hàm main điều khiển tiến trình render
def main():
    """Đọc tệp test.pdf và kết xuất toàn bộ trang thành ảnh PNG lưu ở thư mục hiện tại."""
    pdf_path = Path("test.pdf").resolve()
    output_dir = Path(".").resolve()
    
    if not pdf_path.is_file():
        print(f"Error: test.pdf not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Rendering {pdf_path} using PyMuPDF...")
    images = render_pdf_to_simple_names(pdf_path, output_dir, dpi=300)
    print(f"Successfully rendered {len(images)} pages.")

if __name__ == "__main__":
    main()

