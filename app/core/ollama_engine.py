"""
Module: ollama_engine.py
Nhiệm vụ: Động cơ OCR sử dụng mô hình ngôn ngữ lớn Qwen Vision kết nối thông qua Ollama API.
"""

from pathlib import Path
from time import perf_counter

# pyrefly: ignore [missing-import]
from ollama import Client


# Prompt chính dùng để hướng dẫn Qwen chuyển ảnh tài liệu thành Markdown chất lượng cao
OCR_PROMPT = """
Chuyển đổi hình ảnh trang tài liệu scan này thành định dạng Markdown sạch.

QUY TẮC OCR TUYỆT ĐỐI (ưu tiên cao nhất): Chỉ trích xuất/chép lại chính xác nội dung
văn bản thực sự nhìn thấy trong ảnh. TUYỆT ĐỐI KHÔNG tự trả lời câu hỏi, giải bài tập,
viết đoạn văn/bài văn, viết tiếp phần còn thiếu, suy đoán đáp án, tóm tắt, giải thích
hay bổ sung nội dung mới. Nếu ảnh là đề thi hoặc phiếu câu hỏi, chỉ chép đề và vùng trả
lời trống. Chỉ xuất đáp án/lời giải khi chính những chữ đó có in rõ trên ảnh hiện tại.
Nếu chữ không rõ, giữ phần đọc chắc chắn; không dùng ngữ cảnh để tự hoàn thành nội dung.

Hãy tuân thủ các quy tắc định dạng và cấu trúc nghiêm ngặt sau:
1. Cấu trúc & Bố cục Tài liệu:
   - Nhận diện và định dạng tiêu đề (sử dụng các cấp độ #, ##, ### phù hợp), đoạn văn, danh sách (có thứ tự và không thứ tự), và khối mã (code block).
   - Giữ đúng luồng đọc tự nhiên. Với bố cục nhiều cột, đọc theo từng cột thay vì đọc ngang qua các cột.
   - Tự động phát hiện và loại bỏ các yếu tố nhiễu như tiêu đề đầu trang (headers), chân trang (footers), số trang, và watermark lặp lại.
   - Mỗi heading và mỗi phương án A., B., C., D. phải nằm trên một dòng riêng. Không ghép nhiều heading/phương án trên cùng dòng và không xuất lệnh `\\hfill`.
   - Loại cả tên ấn phẩm, hashtag và số thứ tự lặp lại ở mép chân trang.

2. Biểu thức Toán học:
   - Nhận diện toàn bộ ký hiệu, công thức, biến số, chỉ số, và phương trình toán học.
   - Bao bọc các biểu thức toán học trong dòng bằng một cặp dấu đô-la (`$...$`) và các phương trình độc lập bằng cặp dấu đô-la kép (`$$...$$`).
   - Sử dụng cú pháp LaTeX tiêu chuẩn (ví dụ: các phân số dùng `\frac{num}{den}`, các chữ cái Hy Lạp, ký hiệu toán học).
   - Đảm bảo mỗi công thức toán học có cặp dấu đô-la đóng/mở riêng biệt và chính xác. Không gộp văn bản thường, dấu câu, hoặc nhãn danh sách vào trong dấu đô-la.

3. Bảng biểu & Hình ảnh:
   - Chuyển đổi các bảng biểu đơn giản thành định dạng bảng Markdown tiêu chuẩn.
   - Với các bảng phức tạp (có gộp dòng/gộp cột), sử dụng thẻ HTML `<table>` để biểu diễn chính xác cấu trúc.
   - Chỉ dùng `image_placeholder` cho ảnh chụp, biểu đồ, logo hoặc hình minh họa chủ yếu là đồ họa.
   - Hộp văn bản, callout, ghi chú có khung, biểu mẫu và các nút sơ đồ có chữ BẮT BUỘC được chép đầy đủ thành Markdown; tuyệt đối không thay chữ đọc được bằng placeholder ảnh.
   - Nếu ảnh thật có nhãn hoặc chữ quan trọng, chèn một thẻ ảnh rồi chép phần chữ nhìn thấy ngay bên dưới.

4. Nguyên tắc chung:
   - Giữ nguyên văn bản gốc, bảo toàn ngôn ngữ (tiếng Việt, tiếng Anh, v.v.) và chính tả.
   - Soi chữ nghệ thuật/font cách điệu theo từng ký tự, đặc biệt là dấu tiếng Việt và các chữ dễ nhầm. Chỉ dùng ngữ cảnh để phân biệt nét chữ thực sự nhìn thấy, không được sáng tác chữ mới.
   - Không tóm tắt, giải thích, hoặc viết thêm lời dẫn giải.
   - Chỉ trả về duy nhất nội dung văn bản Markdown thô. Không bọc kết quả trong các khối mã ```markdown.
""".strip()


# Lớp định nghĩa động cơ OllamaQwenEngine để quản lý kết nối và yêu cầu OCR tới Ollama
class OllamaQwenEngine:

    # Hàm khởi tạo động cơ
    def __init__(
        self,
        model="qwen3.5:4b",
        host="http://localhost:11434",
    ):
        self.model = model

        self.client = Client(
            host=host
        )

    # Hàm kiểm tra kết nối tới dịch vụ Ollama local
    def check_connection(self):
        """Kiểm tra kết nối tới máy chủ Ollama bằng cách liệt kê danh sách mô hình."""
        self.client.list()
        return True

    # Hàm chính thực hiện OCR cho hình ảnh và trả về văn bản Markdown sạch
    def ocr_image(
        self,
        image_path: str | Path,
    ) -> str:
        """
        Nhận diện chữ từ ảnh bằng cách gọi API của Ollama, truyền ảnh kèm prompt hệ thống.
        """

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

    # Hàm OCR kèm đo đạc chi tiết chỉ số thời gian và Token sử dụng
    def ocr_image_with_metrics(self, image_path: str | Path) -> tuple[str, dict[str, float | int | None]]:
        """
        Thực hiện OCR hình ảnh và trả về các phép đo thời gian/thống kê token của máy chủ Ollama.
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

        # Trích xuất thời gian dạng nano-giây chuyển sang giây
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

    # Hàm yêu cầu Ollama giải phóng mô hình khỏi bộ nhớ GPU/VRAM
    def unload(self) -> None:
        """Giải phóng mô hình khỏi Ollama để giải phóng tài nguyên GPU."""
        self.client.generate(
            model=self.model,
            prompt="",
            stream=False,
            keep_alive=0,
        )

    # Hàm tĩnh làm sạch văn bản Markdown đầu ra từ Qwen Vision
    @staticmethod
    def clean(text: str) -> str:
        """
        Làm sạch kết quả: loại bỏ các khối mã ```markdown, thực thể khoảng trắng HTML, 
        và sửa lỗi bao bọc toán học kéo dài qua nhiều lựa chọn trắc nghiệm.
        """
        import re
        text = text.strip()

        if text.startswith("```markdown"):
            text = text[len("```markdown"):]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Dọn dẹp các thực thể khoảng trắng HTML
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;nbsp;", " ")
        from app.core.math_cleanup import normalize_answer_math
        return normalize_answer_math(text)
