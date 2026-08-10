from pathlib import Path
from time import perf_counter

# pyrefly: ignore [missing-import]
from ollama import Client


OCR_PROMPT = """
Chuyển đổi hình ảnh trang tài liệu scan này thành định dạng Markdown sạch.

Hãy tuân thủ các quy tắc định dạng và cấu trúc nghiêm ngặt sau:
1. Cấu trúc & Bố cục Tài liệu:
   - Nhận diện và định dạng tiêu đề (sử dụng các cấp độ #, ##, ### phù hợp), đoạn văn, danh sách (có thứ tự và không thứ tự), và khối mã (code block).
   - Giữ đúng luồng đọc tự nhiên. Với bố cục nhiều cột, đọc theo từng cột thay vì đọc ngang qua các cột.
   - Tự động phát hiện và loại bỏ các yếu tố nhiễu như tiêu đề đầu trang (headers), chân trang (footers), số trang, và watermark lặp lại.

2. Biểu thức Toán học:
   - Nhận diện toàn bộ ký hiệu, công thức, biến số, chỉ số, và phương trình toán học.
   - Bao bọc các biểu thức toán học trong dòng bằng một cặp dấu đô-la (`$...$`) và các phương trình độc lập bằng cặp dấu đô-la kép (`$$...$$`).
   - Sử dụng cú pháp LaTeX tiêu chuẩn (ví dụ: các phân số dùng `\frac{num}{den}`, các chữ cái Hy Lạp, ký hiệu toán học).
   - Đảm bảo mỗi công thức toán học có cặp dấu đô-la đóng/mở riêng biệt và chính xác. Không gộp văn bản thường, dấu câu, hoặc nhãn danh sách vào trong dấu đô-la.

3. Bảng biểu & Hình ảnh:
   - Chuyển đổi các bảng biểu đơn giản thành định dạng bảng Markdown tiêu chuẩn.
   - Với các bảng phức tạp (có gộp dòng/gộp cột), sử dụng thẻ HTML `<table>` để biểu diễn chính xác cấu trúc.
   - Với hình vẽ, sơ đồ hoặc minh họa, biểu diễn bằng thẻ ảnh Markdown: `![Mô tả hình ảnh](image_placeholder.png)`.

4. Nguyên tắc chung:
   - Giữ nguyên văn bản gốc, bảo toàn ngôn ngữ (tiếng Việt, tiếng Anh, v.v.) và chính tả.
   - Không tóm tắt, giải thích, hoặc viết thêm lời dẫn giải.
   - Chỉ trả về duy nhất nội dung văn bản Markdown thô. Không bọc kết quả trong các khối mã ```markdown.
""".strip()
# OCR_PROMPT = """
# Chuyển toàn bộ nội dung nhìn thấy trong ảnh tài liệu scan thành Markdown.

# Yêu cầu về công thức toán học (BẮT BUỘC):
# - Tất cả biểu thức toán học, ký hiệu, biến số (ví dụ: x, y, a, b), chỉ số (trên/dưới), phân số, căn thức, tích phân, vector hay ma trận phải được bao bọc bằng định dạng LaTeX.
# - Sử dụng dấu $...$ cho công thức nằm trong cùng dòng văn bản (ví dụ: $f(x) = ax^2 + bx + c$, vector $\\vec{a}$).
# - Sử dụng dấu $$...$$ đặt ở dòng riêng biệt cho các phương trình lớn hoặc công thức độc lập.
# - Sử dụng chính xác các ký hiệu toán học tiêu chuẩn của LaTeX (như \\sin, \\cos, \\pi, \\alpha, \\beta, \\rightarrow, \\cap, \\cup, \\emptyset, \\vec{a}).

# Yêu cầu về bảng biểu (BẮT BUỘC):
# - Nếu bảng có ô gộp (rowspan/colspan), tiêu đề nhiều tầng, hoặc cấu trúc lồng nhau: chuyển thành bảng HTML (<table><tr><td>...</td></tr></table>).
# - Nếu bảng đơn giản, không ô gộp: dùng bảng Markdown (| cột | cột |).

# Yêu cầu về số trang / watermark:
# - Nếu thấy số trang hoặc watermark lặp lại ở đầu/cuối trang, bọc riêng trong <page_number>...</page_number> hoặc <watermark>...</watermark>, không để lẫn vào đoạn văn chính.

# Yêu cầu chung:
# - Sao chép chính xác nội dung, giữ nguyên tiếng Việt và thứ tự đọc tự nhiên.
# - Giữ tiêu đề, đoạn văn, danh sách và bảng biểu.
# - Không được đoán hoặc suy diễn nội dung không nhìn rõ trong ảnh.
# - Không tự giải thích công thức, không viết thêm lời dẫn giải, không sửa đổi chính tả gốc.
# - Nếu trang trống hoặc không đọc được, trả về chuỗi rỗng.
# - Chỉ trả về duy nhất nội dung văn bản Markdown, không bọc trong dấu ```.
# """.strip()

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

    def ocr_image_with_metrics(self, image_path: str | Path) -> tuple[str, dict[str, float | int | None]]:
        """OCR one image and expose Ollama's server timing/token counters.

        ``host_overhead_seconds`` includes client-side image encoding and local HTTP
        overhead; Ollama does not expose those phases separately.
        """
        image_path = Path(image_path).resolve()
        started_at = perf_counter()
        response = self.client.generate(
            model=self.model,
            prompt=OCR_PROMPT,
            images=[str(image_path)],
            think=False,
            stream=False,
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
            keep_alive="10m",
        )
        wall_seconds = perf_counter() - started_at

        def seconds(name: str) -> float | None:
            value = getattr(response, name, None)
            return None if value is None else value / 1_000_000_000

        load_seconds = seconds("load_duration")
        prompt_eval_seconds = seconds("prompt_eval_duration")
        eval_seconds = seconds("eval_duration")
        known_seconds = sum(value for value in (load_seconds, prompt_eval_seconds, eval_seconds) if value is not None)
        return self.clean(response.response), {
            "wall_seconds": wall_seconds,
            "load_seconds": load_seconds,
            "prompt_eval_seconds": prompt_eval_seconds,
            "eval_seconds": eval_seconds,
            "host_overhead_seconds": max(0.0, wall_seconds - known_seconds),
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "generated_tokens": getattr(response, "eval_count", None),
        }

    def unload(self) -> None:
        """Request Ollama to evict the model so the next run measures a cold load."""
        self.client.generate(
            model=self.model,
            prompt="",
            stream=False,
            keep_alive=0,
        )

    @staticmethod
    def clean(text: str) -> str:
        import re
        text = text.strip()

        if text.startswith("```markdown"):
            text = text[len("```markdown"):]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Clean up HTML space entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;nbsp;", " ")

        # Fix math block spanning multiple options (e.g. A. $... B. ...$)
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
