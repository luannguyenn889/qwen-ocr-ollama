# Qwen OCR & Ollama Project

Dự án này sử dụng mô hình thị giác (Vision Language Model) **Qwen** chạy local thông qua **Ollama** để nhận diện ký tự quang học (OCR) các tài liệu tiếng Việt dạng hình ảnh và PDF sang định dạng Markdown chuẩn.

---

## I. Hai cách thiết lập dịch vụ Ollama (Ollama Server)

Để phục vụ OCR, bạn cần có một Ollama Server chạy model `qwen3.5:4b`. Dự án hỗ trợ 2 cách thiết lập sau (chọn 1 trong 2 cách):

---

### Cách 1: Thiết lập trực tiếp trên máy thật (Khuyên dùng khi Dev/Debug)
*   **Đặc điểm:** Tận dụng tối đa phần cứng/GPU của máy thật, dễ chạy trực tiếp mà không cần cài đặt Docker.
*   **Các bước thực hiện:**
    1. Tải và cài đặt Ollama từ trang chủ: [https://ollama.com](https://ollama.com).
    2. Đảm bảo ứng dụng Ollama đã được mở và đang chạy ngầm dưới khay hệ thống.
    3. Mở Terminal/CMD và tải về mô hình Qwen phục vụ cho OCR:
       ```bash
       ollama pull qwen3.5:4b
       ```

---

### Cách 2: Thiết lập thông qua Docker (Đóng gói sẵn Model - Phù hợp Deploy/Chuyển giao offline)
*   **Đặc điểm:** Không cần cài đặt Ollama thủ công, model `qwen3.5:4b` được đóng gói sẵn hoàn toàn bên trong Docker Image (chạy được offline trên các máy khác ngay lập tức không cần tải lại).
*   **Các bước thực hiện:**
    1. Yêu cầu máy đã cài đặt **Docker** và **Docker Desktop**.
    2. **Build Docker Image** (Tự động kéo và đóng gói sẵn model):
       ```bash
       docker build -t local-ollama-qwen .
       ```
    3. **Khởi chạy container**:
       ```bash
       docker run -d -p 11434:11434 --name my-ollama local-ollama-qwen
       ```
       *(Lưu ý: Nếu sử dụng cách này, hãy tắt ứng dụng Ollama trên máy thật nếu đang bật để tránh xung đột cổng `11434`).*

---

## II. Các bước thiết lập môi trường Python chạy Code (Client)

Dù bạn chọn cách chạy Ollama nào ở trên (Cách 1 hay Cách 2), bạn vẫn cần cấu hình môi trường Python trên máy thật để chạy mã nguồn dự án:

1. Đảm bảo máy bạn đã cài đặt **Python 3.10 trở lên**.
2. Mở Terminal tại thư mục gốc của dự án (`qwen-ocr-ollama`) và chạy lần lượt các lệnh:

### Bước 1: Khởi tạo môi trường ảo (Virtual Environment)
```bash
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo
*   **Trên Windows (PowerShell):**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
*   **Trên Windows (CMD):**
    ```cmd
    .\.venv\Scripts\activate.bat
    ```
*   **Trên macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```
*(Khi kích hoạt thành công, bạn sẽ thấy chữ `(.venv)` xuất hiện ở đầu dòng lệnh).*

### Bước 3: Cài đặt các thư viện Python
```bash
pip install -r requirements.txt
```

---

## III. Hướng dẫn chạy các file kiểm thử (Test)

Khi Ollama Server đã khởi chạy và môi trường ảo Python đã được kích hoạt, bạn tiến hành chạy thử các tệp kiểm thử để thực hiện OCR:

### 1. Kiểm tra kết nối Ollama cơ bản
Kiểm tra xem Python đã giao tiếp được với Ollama Server cục bộ (hoặc Docker) chưa:
```bash
python tests/test_ollama.py
```

### 2. Chạy thử OCR hình ảnh đơn lẻ
Nhận diện hình ảnh mẫu `samples/images/test.jpg` và xuất kết quả Markdown ra console cùng tệp tin `output/test_result.md`:
```bash
python tests/test_ocr_image.py
```

### 3. Chạy thử render PDF thành ảnh
Chuyển đổi file PDF mẫu thành các trang ảnh riêng biệt trong thư mục `output/test_pdf_renderer_output/`:
```bash
python tests/test_pdf_renderer.py
```

### 4. Chạy thử OCR toàn bộ tệp PDF
Tách các trang PDF, OCR từng trang rồi gộp chung kết quả vào file Markdown hoàn chỉnh tại `output/test.md`:
```bash
python tests/test_full_pdf.py
```

### 5. Chạy Benchmark đánh giá hình ảnh
Đo lường thời gian xử lý, RAM, VRAM đỉnh và so sánh độ chính xác của 10 ảnh mẫu:
```bash
python tests/test_benchmark_images.py
```

---


## V. Cấu trúc thư mục dự án

```text
qwen-ocr-ollama/
│
├── .venv/                      # Môi trường ảo Python (được ignore trên Git)
├── app/
│   └── core/                   # Chứa logic cốt lõi xử lý OCR, render PDF
│       ├── ollama_engine.py    # Giao tiếp với Ollama API
│       ├── pdf_rerender.py     # Render PDF sang ảnh tạm thời
│       └── pdf_ocr.py          # Điều phối quy trình OCR toàn bộ tệp PDF
│
├── samples/                    # Dữ liệu thử nghiệm
│   ├── images/                 # Ảnh đầu vào cho kiểm thử
│   └── pdfs/                   # File PDF đầu vào cho kiểm thử
│
├── output/                     # Thư mục chứa kết quả đầu ra của OCR & Benchmark
│
├── tests/                      # Thư mục chứa các file script chạy thử nghiệm
│   ├── guide.txt               # Hướng dẫn nhanh chạy các test
│   ├── test_ollama.py
│   ├── test_ocr_image.py
│   ├── test_pdf.py
│   └── test_benchmark_images.py
│
├── .gitignore                  # Chỉ định các thư mục/file không đẩy lên Git
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Hướng dẫn cài đặt và sử dụng dự án
```
