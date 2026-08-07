from pathlib import Path
from time import perf_counter

from ollama import Client


MODEL = "qwen3.5:4b"

PROMPT = """
Bạn là hệ thống OCR chính xác cao cho tài liệu scan.

Nhiệm vụ: chép lại toàn bộ chữ nhìn thấy trong ảnh thành Markdown.

Quy tắc bắt buộc:
- Chỉ chép nội dung thực sự nhìn thấy; không suy diễn hoặc bổ sung.
- Giữ nguyên ngôn ngữ, dấu tiếng Việt, chính tả, số, ký hiệu và xuống dòng.
- Nếu chữ hoặc ký hiệu không đọc được, ghi `[không rõ]`; không đoán.
- Giữ thứ tự đọc tự nhiên. Với tài liệu nhiều cột: đọc hết cột trái, sau đó mới đến cột phải.
- Giữ tiêu đề, các cấp heading, đoạn văn, danh sách, chú thích, đầu trang và cuối trang.
- Chuyển bảng sang Markdown. Nếu bảng phức tạp, có gộp ô hoặc Markdown làm mất cấu trúc, dùng HTML `<table>`.
- Chuyển công thức toán học sang LaTex, dùng `$...$` hoặc `$$...$$` khi phù hợp.
- Không tóm tắt, giải thích, dịch, sửa chính tả hay mô tả ảnh.
- Không bọc kết quả trong khối mã Markdown.
- Chỉ trả về nội dung Markdown.
""".strip()


def main():
    image_path = Path(
        "samples/images/test.jpg"
    ).resolve()

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
    output_file = output_dir / "test_result.md"
    output_file.write_text("".join(ocr_result), encoding="utf-8")
    print(f"Đã lưu kết quả tại: {output_file.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
