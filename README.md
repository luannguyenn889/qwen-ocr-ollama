# Qwen OCR & Ollama — Chuyển đổi PDF Scan sang Markdown Offline

Ứng dụng nhận diện chữ (OCR) tài liệu PDF scan sang định dạng **Markdown chuẩn** chạy hoàn toàn **Offline**, sử dụng mô hình thị giác ngôn ngữ lớn **Qwen2.5-VL** thông qua **Ollama**.

---

## 🌟 Tính năng nổi bật

* 🎯 **Độ chính xác cao với Tiếng Việt**: Nhận diện chuẩn xác tài liệu scan tiếng Việt, chữ in mờ, tài liệu lưu trữ cũ và công thức toán học.
* 📊 **Smart Table Merger**: Tự động phát hiện, khử lặp header và nối các bảng biểu bị ngắt đoạn qua nhiều trang PDF.
* 🖼️ **Trích xuất ảnh & Loại bỏ con dấu/chữ ký**: Tự động tách ảnh minh họa nội dung vào thư mục `images/` và lọc bỏ các khối con dấu/chữ ký nhạy cảm.
* 🚫 **Tự động lọc trang trắng (Blank Page Detection)**: Tự động bóc viền scan, khử nhiễu mép và bỏ qua các trang trắng để tăng tốc xử lý.
* 🖥️ **Giao diện GUI Tkinter trực quan**: Hỗ trợ chọn tệp đơn lẻ hoặc quét cả thư mục, hiển thị 2 thanh tiến trình (File & Page) và bảng log thời gian thực.
* 🔒 **100% Offline & Bảo mật**: Xử lý hoàn toàn tại máy local, không gửi dữ liệu ra ngoài Internet.

> 📖 **Xem chi tiết quy trình xử lý chuyên sâu:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 💻 Yêu cầu hệ thống

* **Hệ điều hành:** Windows 10/11 (64-bit) hoặc Linux.
* **CPU:** Tối thiểu 4 lõi / 8 luồng.
* **RAM:** Tối thiểu 16 GB.
* **GPU:** Khuyến nghị NVIDIA GPU có **6 GB VRAM trở lên** (ví dụ: RTX 3050/4050/3060...).
* **Phần mềm:** Python 3.10+ và [Ollama](https://ollama.com).

---

## 🚀 Hướng dẫn Cài đặt Nhanh

### Bước 1: Khởi chạy Ollama & Tải Model
1. Cài đặt Ollama từ [ollama.com](https://ollama.com) và bật ứng dụng.
2. Mở Terminal và tải model thị giác:
   ```bash
   # Khuyến nghị cho GPU RTX (VRAM >= 6GB)
   ollama pull qwen2.5vl:7b

   # Hoặc bản nhẹ cho máy ít VRAM / chạy CPU
   ollama pull qwen3.5:4b
   ```

### Bước 2: Thiết lập môi trường Python
Mở Terminal tại thư mục dự án và chạy:
```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 📖 Hướng dẫn Sử dụng

### 1. Khởi chạy Giao diện Đồ họa (GUI)
Giao diện Tkinter thân thiện, hỗ trợ chọn thư mục PDF đầu vào và thư mục lưu Markdown đầu ra:
```powershell
python run_gui.py
```

### 2. Xử lý Hàng loạt qua Dòng lệnh (CLI Batch)
Tự động quét tất cả file PDF trong thư mục `PDF/` và xuất kết quả sang `OCR/`:
```powershell
python batch_ocr.py --input PDF --output OCR --model qwen2.5vl:7b --workers 1
```
*(Các mức nhạy lọc trang trắng: `--blank-sensitivity safe | standard | aggressive`)*.

### 3. Kiểm tra Tính hợp lệ của Markdown
```powershell
python ocr_validator.py
```

### 4. Đo lường Độ chính xác so với Bản chuẩn (Accuracy Benchmark)
So sánh kết quả OCR với bộ dữ liệu chuẩn (Ground Truth) để tính tỷ lệ chính xác (CER, WER, Table F1):
```powershell
python evaluate_accuracy.py --output OCR --ground-truth samples/ground_truth
```

### 5. Chạy Toàn bộ Kiểm thử Tự động (Unit Tests)
```powershell
pytest tests/
```

---

## 🛠️ Khắc phục sự cố nhanh (Troubleshooting)

* **Cảnh báo DLL / PyTorch khi import Paddle trên Windows (`WinError 127`):**
  * Cài đặt bộ [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
  * Hoặc tắt nạp module layout nếu chỉ dùng Qwen-VL thuần túy:
    ```powershell
    $env:QWEN_OCR_DISABLE_LAYOUT="1"
    python run_gui.py
    ```
* **Cảnh báo `No ccache found`:**
  * Đây chỉ là cảnh báo compiler cache của thư viện Paddle, hoàn toàn không ảnh hưởng đến quá trình OCR.

---

## 📁 Cấu trúc Thư mục Dự án

```text
qwen-ocr-ollama/
├── app/
│   ├── core/                   # Logic xử lý OCR, trích xuất ảnh, chuẩn hóa Markdown
│   │   ├── batch_ocr.py        # Pipeline OCR hàng loạt chính
│   │   ├── block_assembler.py  # Ghép nối khối chữ, bảng và ảnh
│   │   ├── layout_detector.py  # Nhận diện bố cục & bảng biểu
│   │   ├── markdown_normalizer.py # Chuẩn hóa cú pháp Markdown & khử lặp
│   │   ├── quality_gate.py     # Đánh giá chất lượng trang OCR
│   │   ├── vietnamese_lexicon.json # Từ điển tiếng Việt 2.4MB
│   │   └── vietnamese_spell_corrector.py # Bộ kiểm tra chính tả tiếng Việt
│   └── gui/
│       └── run_gui.py          # Giao diện người dùng Tkinter
├── PDF/                        # Thư mục chứa file PDF đầu vào mặc định
├── OCR/                        # Thư mục chứa file Markdown đầu ra mặc định
├── samples/                    # Bộ Testcases & Ground Truth mẫu (10 kịch bản scan)
├── tests/                      # Bộ 225 bài kiểm thử tự động
├── run_gui.py                  # Entrypoint khởi chạy GUI
├── batch_ocr.py                # Entrypoint chạy Batch CLI
├── evaluate_accuracy.py        # Công cụ chấm điểm độ chính xác OCR vs Ground Truth
├── ocr_validator.py            # Công cụ kiểm tra hợp lệ file Markdown
├── pytest.ini                  # Cấu hình kiểm thử pytest
├── requirements.txt            # Danh sách thư viện Python
├── ARCHITECTURE.md             # Chi tiết kỹ thuật & giải thuật chuyên sâu
└── README.md                   # Hướng dẫn tổng quan (File này)
```
