"""
Module: step7_8_ocr_stitch.py
Nhiệm vụ: OCR từng trang ảnh tạm thời từ PDF bằng Ollama Qwen và ghép các trang thành một file Markdown duy nhất.
Quy trình:
  1. Quét toàn bộ ảnh dạng page_*.png trong thư mục hiện tại.
  2. OCR từng ảnh bằng mô hình Qwen.
  3. Ghép nội dung các trang và phân tách bằng bình luận <!-- Page X --> ghi vào file test.md.
"""

import sys
from pathlib import Path
from time import perf_counter
# pyrefly: ignore [missing-import]
from ollama import Client
from app.core.batch_ocr import PROMPT

MODEL = "qwen3.5:4b"

# Hàm làm sạch sơ bộ văn bản Markdown đầu ra
def clean_markdown(text: str) -> str:
    """Làm sạch các rào chắn mã Markdown dư thừa, khoảng trắng HTML và sửa lại khối công thức toán trắc nghiệm."""
    import re
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Loại bỏ khoảng trắng HTML
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;nbsp;", " ")
    
    # Phân tách công thức bị gộp của các lựa chọn trắc nghiệm
    def repl(match):
        content = match.group(1)
        parts = re.split(r"(\s+[B-D]\.\s+)", content)
        if len(parts) > 1:
            new_parts = []
            for part in parts:
                if re.match(r"^\s+[B-D]\.\s+$", part):
                    new_parts.append(part)
                else:
                    stripped = part.strip()
                    if stripped:
                        new_parts.append(f"${stripped}$")
                    else:
                        new_parts.append(part)
            return "".join(new_parts)
        return match.group(0)

    text = re.sub(r"\$([^$\n]+)\$", repl, text)
    return text


# Hàm main điều khiển tiến trình ghép nối OCR
def main():
    """Quét các ảnh page_*.png, chạy nhận diện OCR và ghi ghép nối thành file test.md."""
    root_dir = Path(".").resolve()
    # Tìm các tệp page_X.png, sắp xếp theo thứ tự số trang tăng dần
    image_paths = sorted(
        list(root_dir.glob("page_*.png")),
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    if not image_paths:
        print("Error: No page_*.png files found in the current directory.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(image_paths)} pages to OCR.")
    
    client = Client(host="http://localhost:11434")
    output_path = Path("test.md").resolve()
    
    with output_path.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(image_paths):
            page_num = idx + 1
            print(f"OCR'ing page {page_num}/{len(image_paths)}: {img_path.name}...", flush=True)
            started_at = perf_counter()
            
            response = client.generate(
                model=MODEL,
                prompt=PROMPT,
                images=[str(img_path)],
                think=False,
                stream=False,
                options={
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                },
                keep_alive="10m",
            )
            
            page_markdown = clean_markdown(response.response)
            elapsed = perf_counter() - started_at
            print(f"Page {page_num} completed in {elapsed:.1f} seconds.", flush=True)
            
            # Ghép nối có phân tách trang
            f.write(f"<!-- Page {page_num} -->\n\n")
            f.write(page_markdown)
            f.write("\n\n")
            
    print(f"Stitched PDF OCR output written to {output_path}", flush=True)

if __name__ == "__main__":
    main()
