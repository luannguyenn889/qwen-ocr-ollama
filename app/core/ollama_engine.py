from pathlib import Path

# pyrefly: ignore [missing-import]
from ollama import Client


OCR_PROMPT = """
Chuyển toàn bộ nội dung nhìn thấy trong ảnh tài liệu scan thành Markdown.

Yêu cầu:
- Sao chép chính xác nội dung.
- Giữ nguyên tiếng Việt.
- Giữ đúng thứ tự đọc.
- Giữ heading, đoạn văn và danh sách.
- Chuyển bảng sang Markdown hoặc HTML.
- Công thức chuyển sang LaTeX.
- Không tóm tắt.
- Không giải thích.
- Không sửa nội dung gốc.
- Không thêm thông tin.
- Chỉ trả Markdown.
""".strip()


class OllamaQwenEngine:

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