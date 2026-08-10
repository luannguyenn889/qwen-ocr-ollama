# Nhiệm vụ: Kiểm tra quy trình OCR một file ảnh đơn lẻ (samples/images/test.jpg) qua thư viện Ollama, in trực tiếp kết quả Markdown theo thời gian thực (stream) và lưu vào thư mục output/test_result.md.

from pathlib import Path
from time import perf_counter

# pyrefly: ignore [missing-import]
from ollama import Client


MODEL = "qwen3.5:4b"

PROMPT = """
Chuyển toàn bộ nội dung nhìn thấy trong ảnh tài liệu này thành Markdown.

Yêu cầu bắt buộc:
- Sao chép chính xác nội dung nhìn thấy.
- Giữ nguyên tiếng Việt và dấu tiếng Việt.
- Giữ đúng thứ tự đọc.
- Giữ tiêu đề và các cấp tiêu đề.
- Giữ đoạn văn và danh sách.
- Chuyển bảng sang Markdown hoặc HTML.
- Công thức toán chuyển sang LaTeX.
- Không tóm tắt.
- Không giải thích.
- Không sửa chính tả của tài liệu.
- Không thêm thông tin không xuất hiện trong ảnh.
- Chỉ trả về Markdown.
""".strip()


def main():
    # image_path = Path(
    #     "samples/images/test.jpg"
    # ).resolve()
    image_path = Path("samples/images/testparagraph.png").resolve()
    # image_path = Path("samples/images/devan12.jpeg").resolve()

    print(f"[1/3] Đang kiểm tra ảnh: {image_path}", flush=True)
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {image_path}")
    print(f"      Kích thước ảnh: {image_path.stat().st_size:,} bytes", flush=True)

    print("[2/3] Đang kết nối Ollama tại http://localhost:11434...", flush=True)
    client = Client(
        host="http://localhost:11434"
    )

    print(f"[3/3] Đang OCR bằng {MODEL}. Token sẽ hiện bên dưới khi model bắt đầu trả lời:", flush=True)
    started_at = perf_counter()
    stream = client.generate(
        model=MODEL,
        prompt=PROMPT,
        images=[str(image_path)],
        think=False,
        stream=True,
        options={
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        keep_alive="10m",
    )

    ocr_result = []
    for chunk in stream:
        content = chunk.response
        print(content, end="", flush=True)
        ocr_result.append(content)

    elapsed = perf_counter() - started_at
    print(f"\n\nHoàn tất OCR trong {elapsed:.1f} giây.", flush=True)

    # Lưu kết quả ra file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "test_result3.md"
    output_file.write_text("".join(ocr_result), encoding="utf-8")
    print(f"Đã lưu kết quả tại: {output_file.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
