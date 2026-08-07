"""
Module: ollama_engine
Nhiệm vụ: Quản lý kết nối và thực hiện truy vấn với Ollama Server, sử dụng mô hình Qwen-VL (hoặc Qwen2.5/Qwen3.5) để nhận diện văn bản (OCR) từ hình ảnh.
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
from ollama import Client


# Prompt hướng dẫn mô hình chuyển đổi ảnh tài liệu thành định dạng Markdown chuẩn
OCR_PROMPT = """
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


class OllamaQwenEngine:
    """
    Lớp Engine điều phối giao tiếp với Ollama API.
    Nhiệm vụ chính: Gửi ảnh kèm prompt OCR và dọn dẹp kết quả Markdown trả về.
    """

    def __init__(
        self,
        model="qwen3.5:4b",
        host="http://localhost:11434",
    ):
        self.model = model

        self.client = Client(
            host=host
        )

    def check_connection(self):
        self.client.list()
        return True

    def ocr_image(
        self,
        image_path: str | Path,
    ) -> str:

        image_path = Path(
            image_path
        ).resolve()

        response = self.client.generate(
            model=self.model,
            prompt=OCR_PROMPT,
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

        return self.clean(
            response.response
        )

    @staticmethod
    def clean(text: str) -> str:

        text = text.strip()

        if text.startswith("```markdown"):
            text = text[len("```markdown"):]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()
