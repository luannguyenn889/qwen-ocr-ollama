# Qwen OCR & Ollama Project

Dự án này sử dụng mô hình thị giác (Vision Language Model) **Qwen** chạy local thông qua **Ollama** để nhận diện ký tự quang học (OCR) các tài liệu tiếng Việt dạng hình ảnh và PDF sang định dạng Markdown chuẩn.

---

## I. Yêu cầu hệ thống (Prerequisites)

Trước khi chạy dự án trên máy mới, hãy chuẩn bị đầy đủ các thành phần sau:

1. **Python 3.10 trở lên** (Đã được cấu hình PATH).
2. **Ollama**:
   - Tải và cài đặt Ollama từ trang chủ: [https://ollama.com](https://ollama.com)
   - Đảm bảo ứng dụng Ollama đang chạy ngầm trên máy.
3. **Mô hình Qwen**:
   - Mở Terminal/CMD và tải về mô hình Qwen phục vụ cho OCR:
     ```bash
     ollama pull qwen3.5:4b
     ```

---

## II. Các bước thiết lập khi tải (pull) dự án về máy mới

Mở Terminal/PowerShell tại thư mục gốc của dự án (`qwen-ocr-ollama`) và thực hiện các bước sau:

### Bước 1: Khởi tạo môi trường ảo (Virtual Environment)
```bash
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo
- **Trên Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Trên Windows (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Trên macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
*(Khi kích hoạt thành công, bạn sẽ thấy chữ `(.venv)` xuất hiện ở đầu dòng lệnh Terminal).*

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## III. Hướng dẫn chạy thử nghiệm các tệp kiểm thử (Test)

Sau khi môi trường ảo đã được kích hoạt, bạn có thể chạy thử các kịch bản kiểm thử sau:

### 1. Kiểm tra kết nối Ollama cơ bản
Kiểm tra xem Python đã giao tiếp được với Ollama Server cục bộ chưa:
```bash
python tests/test_ollama.py
```

### 2. Chạy thử OCR hình ảnh đơn lẻ
Nhận diện hình ảnh mẫu `samples/images/test.jpg` và xuất kết quả Markdown ra console và tệp tin `output/test_result.md`:
```bash
python tests/test_ocr_image.py
```

### 3. Chạy thử OCR tệp PDF
Tự động tách các trang của tệp PDF `samples/pdfs/test.pdf` thành hình ảnh tạm thời rồi tiến hành OCR toàn bộ các trang, kết quả cuối cùng sẽ được gộp chung vào một file Markdown nằm trong thư mục `output/`:
```bash
python tests/test_pdf.py
```

### 4. Chạy Benchmark đánh giá hình ảnh
Đo lường thời gian xử lý, dung lượng RAM, dung lượng VRAM đỉnh và kết quả đầu ra của bộ 10 ảnh benchmark:
```bash
python tests/test_benchmark_images.py
```

---

## IV. Cấu trúc thư mục dự án

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
