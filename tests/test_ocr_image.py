from pathlib import Path

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
    image_path = Path(
        "samples/images/test.png"
    ).resolve()

    client = Client(
        host="http://localhost:11434"
    )

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

    print(response.response)


if __name__ == "__main__":
    main()