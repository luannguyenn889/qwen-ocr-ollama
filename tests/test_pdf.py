# Nhiệm vụ: Kiểm tra quy trình trích xuất trang PDF, render thành ảnh tạm thời, thực hiện OCR từng trang bằng OllamaQwenEngine và in kết quả ra terminal để kiểm tra nhanh.

import sys
import tempfile
from pathlib import Path
from time import perf_counter

# Thêm thư mục gốc của dự án vào sys.path để Python tìm thấy package 'app'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.core.ollama_engine import (
    OllamaQwenEngine,
)

from app.core.pdf_rerender import (
    render_pdf,
)

engine = OllamaQwenEngine()

pdf = Path(
    "samples/pdfs/test.pdf"
).resolve()

print(f"[*] Bắt đầu kiểm tra PDF: {pdf}", flush=True)
if not pdf.is_file():
    print(f"[!] Lỗi: Không tìm thấy file PDF tại {pdf}", flush=True)
    exit(1)

with tempfile.TemporaryDirectory() as tmp:
    print(f"[*] Đang chuyển đổi các trang PDF thành ảnh (DPI=200)...", flush=True)
    
    for page_number, image_path in render_pdf(
        pdf,
        tmp,
        dpi=200,
    ):
        print(f"\n[+] Đang xử lý Trang {page_number}...", flush=True)
        print(f"    - Đường dẫn ảnh tạm: {image_path}", flush=True)
        print(f"    - Đang thực hiện OCR bằng Qwen (Ollama)...", flush=True)
        
        started_at = perf_counter()
        markdown = engine.ocr_image(
            image_path
        )
        elapsed = perf_counter() - started_at
        
        print(f"    - Hoàn tất OCR trang {page_number} trong {elapsed:.1f} giây. Kết quả:\n", flush=True)
        print("--- KẾT QUẢ ---")
        print(markdown)
        print("---------------\n")