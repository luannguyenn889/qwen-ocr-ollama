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
   - Sử dụng cú pháp LaTeX tiêu chuẩn (ví dụ: các phân số dùng `\\frac{num}{den}`, các chữ cái Hy Lạp, ký hiệu toán học).
   - Đảm bảo mỗi công thức toán học có cặp dấu đô-la đóng/mở riêng biệt và chính xác. Không gộp văn bản thường, dấu câu, hoặc nhãn danh sách vào trong dấu đô-la.

3. Bảng biểu & Hình ảnh:
   - Chuyển đổi các bảng biểu đơn giản thành định dạng bảng Markdown tiêu chuẩn (`| Cột 1 | Cột 2 |`).
   - Với các bảng phức tạp (có gộp dòng/gộp cột, header nhiều tầng, rowspan, colspan), BẮT BUỘC sử dụng thẻ HTML `<table>` (`<tr>`, `<th>`, `<td>`, `rowspan`, `colspan`) để biểu diễn chính xác cấu trúc.
   - Quy tắc nghiêm ngặt cho bảng Markdown:
     * BẮT BUỘC làm phẳng tiêu đề đa cấp thành một dòng header duy nhất (ví dụ: gộp tiêu đề cha và con: "Kết quả thống kê - Đơn vị tính | Kết quả thống kê - Số liệu").
     * BẮT BUỘC đồng nhất số lượng cột: Dòng tiêu đề, dòng gạch ngang phân cách (`:---`) và tất cả các dòng dữ liệu BẮT BUỘC phải có chính xác cùng số lượng cột (N cột), tuyệt đối không lệch cột.
   - Nếu bảng kéo dài qua trang sau, giữ nguyên cấu trúc cột và không tự tạo tiêu đề sai lệch.
   - Chỉ dùng `image_placeholder` cho ảnh chụp, biểu đồ, logo hoặc hình minh họa chủ yếu là đồ họa.
   - Hộp văn bản, callout, ghi chú có khung, biểu mẫu và các nút sơ đồ có chữ BẮT BUỘC được chép đầy đủ thành Markdown; tuyệt đối không thay chữ đọc được bằng placeholder ảnh.
   - Nếu ảnh thật có nhãn hoặc chữ quan trọng, chèn một thẻ ảnh rồi chép phần chữ nhìn thấy ngay bên dưới.

4. Biểu mẫu, Đánh dấu Bút mực, Checkbox & Lựa chọn:
   - Checkbox biểu mẫu: Ô vuông chưa chọn xuất thành `- [ ]` (hoặc `[ ]`); ô đã tích/chọn (đánh dấu ✓, ✗, x, tô đậm) xuất thành `- [x]` (hoặc `[x]`).
   - Khoanh tròn đáp án trắc nghiệm: Khi chữ cái đáp án (A., B., C., D. hoặc A, B, C, D) bị khoanh tròn bằng bút mực, xuất định dạng **(A)** để phân biệt đáp án được chọn.
   - Gạch xóa bút mực: Khi có dòng chữ in bị gạch xóa bằng bút, bao bọc đoạn bị gạch trong cú pháp `~~nội dung bị gạch~~`.
   - Chữ viết tay điền biểu mẫu: Chép chính xác nội dung chữ viết tay tại đúng vị trí điền trên biểu mẫu.

5. Nguyên tắc chung, Ký hiệu, Danh từ riêng & Dấu Tiếng Việt:
   - Giữ nguyên văn bản gốc, bảo toàn ngôn ngữ (tiếng Việt, tiếng Anh, v.v.) và chính tả.
   - Danh từ riêng, Từ địa phương & Cấu trúc âm tiết đặc thù:
     * Tài liệu có thể chứa danh từ riêng, tên địa danh, từ địa phương hoặc tên người dân tộc với các cấu trúc âm tiết ít gặp trong từ điển chuẩn (như các tổ hợp nguyên âm đặc thù, âm đuôi kết thúc bằng `-nh`, `-k`, `-r`, `-p`, `-l`, `-s`, `-ch`).
     * BẮT BUỘC chép trung thực 100% từng ký tự chữ cái và dấu thanh theo đúng hình ảnh nhìn thấy.
     * Đặc biệt chú ý phân biệt nét ký tự ở đuôi chữ (như chữ `h` vs `g`, `n` vs `m`, `k` vs `c`, `t` vs `l`).
     * TUYỆT ĐỐI KHÔNG tự ý sửa từ (autocorrect), không ép về từ vựng phổ thông quen thuộc và không tự thêm/bớt dấu thanh khi ảnh gốc không có.
   - Bảo toàn các ký hiệu đặc biệt, đơn vị đo lường (ví dụ: độ `°C`, kích thước `×`, micro `µ`, cộng trừ `±`, so sánh `≤`, `≥`, `≈`, bản quyền `©`, `®`, `™`, chỉ số `m²`, `m³`, `x₁`).
   - Chú ý đặc biệt đến hệ thống dấu tiếng Việt:
     * Phân biệt chính xác tuyệt đối giữa dấu nặng (.) và dấu sắc ('), dấu hỏi (?) và dấu ngã (~), dấu mũ (â, ê, ô) và dấu móc (ơ, ư), chữ 'đ' và 'd'.
     * Phân biệt chuẩn xác giữa chữ 'n' và 'm' ở cuối từ (ví dụ: 'giản' vs 'giảm', 'kiện' vs 'kiến'), chữ 'có' vs 'cơ'.
     * Đọc trung thực tuyệt đối theo hình ảnh thực tế nhìn thấy, không tự ý suy diễn hoặc thay thế từ ngữ.
   - Soi chữ nghệ thuật/font cách điệu theo từng ký tự. Chỉ dùng ngữ cảnh để phân biệt nét chữ thực sự nhìn thấy, không được sáng tác chữ mới.
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
            keep_alive="30m",
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
            keep_alive="30m",
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

    # Hàm khởi động và nạp sẵn mô hình lên VRAM
    def warmup(self, keep_alive: str = "30m") -> bool:
        """Nạp sẵn mô hình lên VRAM để giảm độ trễ cho request đầu tiên."""
        try:
            self.client.generate(
                model=self.model,
                prompt="",
                stream=False,
                keep_alive=keep_alive,
            )
            return True
        except Exception:
            return False

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

        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\r?\n?", "", text)
        text = re.sub(r"\r?\n?```\s*$", "", text)
        text = text.strip()

        # Dọn dẹp các thực thể khoảng trắng HTML
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;nbsp;", " ")
        from app.core.math_cleanup import normalize_answer_math
        return normalize_answer_math(text)
