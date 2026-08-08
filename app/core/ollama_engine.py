from pathlib import Path

# pyrefly: ignore [missing-import]
from ollama import Client


OCR_PROMPT = """
Chuyển toàn bộ nội dung nhìn thấy trong ảnh tài liệu scan thành Markdown.

Yêu cầu về công thức toán học (BẮT BUỘC):
- Tất cả biểu thức toán học, ký hiệu, biến số (ví dụ: x, y, a, b), chỉ số (trên/dưới), phân số, căn thức, tích phân, vector hay ma trận phải được bao bọc bằng định dạng LaTeX.
- Sử dụng dấu $...$ cho công thức nằm trong cùng dòng văn bản (ví dụ: $f(x) = ax^2 + bx + c$, vector $\vec{a}$).
- Sử dụng dấu $$...$$ đặt ở dòng riêng biệt cho các phương trình lớn hoặc công thức độc lập.
- Sử dụng chính xác các ký hiệu toán học tiêu chuẩn của LaTeX (như \\sin, \\cos, \\pi, \\alpha, \\beta, \\rightarrow, \\cap, \\cup, \\emptyset, \\vec{a}).

Yêu cầu chung:
- Sao chép chính xác nội dung, giữ nguyên tiếng Việt và thứ tự đọc.
- Giữ tiêu đề, đoạn văn, danh sách và bảng biểu.
- Không tự giải thích công thức, không viết thêm lời dẫn giải, không sửa đổi chính tả gốc.
- Chỉ trả về duy nhất nội dung văn bản Markdown.
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