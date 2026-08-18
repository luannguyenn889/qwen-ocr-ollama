"""
Module: ocr_single.py
Nhiệm vụ: Chạy nhận diện OCR cho một tệp hình ảnh đơn lẻ sử dụng mô hình Qwen qua Ollama Client.
"""

import sys
from pathlib import Path
from time import perf_counter
from ollama import Client
from app.core.ollama_engine import OCR_PROMPT

MODEL = "qwen3.5:4b"

PROMPT = OCR_PROMPT

# Hàm main điều khiển tiến trình nhận diện hình ảnh đơn
def main():
    """Đọc ảnh test.png, gửi yêu cầu tới mô hình Qwen qua Ollama và ghi kết quả vào test.md."""
    image_path = Path("test.png").resolve()
    output_path = Path("test.md").resolve()

    if not image_path.is_file():
        print(f"Error: test.png not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"OCR'ing {image_path} with {MODEL}...", flush=True)
    started_at = perf_counter()

    client = Client(host="http://localhost:11434")
    
    response = client.generate(
        model=MODEL,
        prompt=PROMPT,
        images=[str(image_path)],
        think=False,
        stream=False,
        options={
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        keep_alive="10m",
    )

    raw_text = response.response.strip()

    # Làm sạch các rào chắn Markdown nếu có
    clean_text = raw_text
    if clean_text.startswith("```markdown"):
        clean_text = clean_text[len("```markdown"):]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    output_path.write_text(clean_text, encoding="utf-8")
    elapsed = perf_counter() - started_at
    print(f"Completed in {elapsed:.1f} seconds. Output written to {output_path}", flush=True)

if __name__ == "__main__":
    main()
