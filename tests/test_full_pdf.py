import sys
from pathlib import Path
from time import perf_counter

# Cho phép chạy trực tiếp bằng: python tests\test_full_pdf.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ollama_engine import OllamaQwenEngine
from app.core.pdf_ocr import ocr_pdf


PDF_PATH = PROJECT_ROOT / "samples" / "pdfs" / "test.pdf"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    print("=== OCR toàn bộ PDF ===", flush=True)
    print(f"PDF: {PDF_PATH}", flush=True)
    print(f"Thư mục kết quả: {OUTPUT_DIR}", flush=True)

    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF: {PDF_PATH}")

    started_at = perf_counter()
    engine = OllamaQwenEngine()
    markdown_path = ocr_pdf(
        engine=engine,
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        dpi=200,
    )
    elapsed = perf_counter() - started_at

    print(f"\nĐã hoàn tất trong {elapsed:.1f} giây.", flush=True)
    print(f"Kết quả Markdown: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
