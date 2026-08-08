# Nhiệm vụ: Đánh giá và tối ưu hóa thông số DPI bằng cách so sánh hiệu năng của các mức DPI: 180, 200, 250, 300.

import sys
import shutil
from pathlib import Path
from time import perf_counter
from difflib import SequenceMatcher

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import pymupdf
from app.core.pdf_renderer import render_pdf_to_images
from app.core.ollama_engine import OllamaQwenEngine

def calculate_similarity(text1: str, text2: str) -> float:
    # Chuẩn hóa văn bản cơ bản để so sánh chính xác hơn (bỏ khoảng trắng thừa)
    t1 = " ".join(text1.strip().split())
    t2 = " ".join(text2.strip().split())
    return SequenceMatcher(None, t1, t2).ratio() * 100

def run_dpi_benchmark():
    pdf_source = PROJECT_ROOT / "samples" / "pdfs" / "test.pdf"
    gt_file = PROJECT_ROOT / "samples" / "ground_truth" / "test_page_1.md"
    output_dir = PROJECT_ROOT / "output" / "dpi_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_source.is_file():
        print(f"[!] Lỗi: Không tìm thấy file PDF nguồn tại {pdf_source}")
        return
    if not gt_file.is_file():
        print(f"[!] Lỗi: Không tìm thấy file Ground Truth tại {gt_file}")
        return
        
    ground_truth = gt_file.read_text(encoding="utf-8")
    
    # 1. Trích xuất trang 1 của test.pdf để làm file PDF thử nghiệm (sử dụng PID để tránh tranh chấp file tạm)
    import os
    temp_pdf = output_dir / f"temp_page_1_{os.getpid()}.pdf"
    src_doc = pymupdf.open(pdf_source)
    dest_doc = pymupdf.open()
    dest_doc.insert_pdf(src_doc, from_page=0, to_page=0)
    dest_doc.save(temp_pdf)
    dest_doc.close()
    src_doc.close()
    
    dpis = [180, 200, 250, 300]
    results = {}
    engine = OllamaQwenEngine()
    
    print("=== BẮT ĐẦU BENCHMARK TỐI ƯU HÓA DPI ===")
    print(f"File PDF test (1 trang): {temp_pdf.name}")
    print(f"Các mức DPI thử nghiệm: {dpis}\n")
    
    for dpi in dpis:
        print(f"[*] Đang thử nghiệm với DPI = {dpi}...", flush=True)
        dpi_output_dir = output_dir / f"dpi_{dpi}"
        if dpi_output_dir.exists():
            shutil.rmtree(dpi_output_dir)
        dpi_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Đo thời gian render PDF sang ảnh
        t_start_render = perf_counter()
        images = render_pdf_to_images(temp_pdf, dpi_output_dir, dpi=dpi)
        t_render = perf_counter() - t_start_render
        
        image_path = images[0]
        image_size_kb = image_path.stat().st_size / 1024
        
        # Đo thời gian thực hiện OCR bằng Ollama
        t_start_ocr = perf_counter()
        ocr_result = engine.ocr_image(image_path)
        t_ocr = perf_counter() - t_start_ocr
        
        # Tính độ chính xác so với Ground Truth
        similarity = calculate_similarity(ocr_result, ground_truth)
        
        # Lưu lại kết quả
        results[dpi] = {
            "time_render": t_render,
            "time_ocr": t_ocr,
            "total_time": t_render + t_ocr,
            "image_size_kb": image_size_kb,
            "similarity": similarity,
            "ocr_text": ocr_result
        }
        
        print(f"    - Render: {t_render:.2f}s | OCR: {t_ocr:.2f}s | Tổng: {t_render + t_ocr:.2f}s")
        print(f"    - Kích thước ảnh: {image_size_kb:.1f} KB")
        print(f"    - Độ chính xác (so với GT): {similarity:.2f}%\n", flush=True)
        
        # Lưu kết quả OCR của từng DPI ra file để tham chiếu
        ocr_out_file = dpi_output_dir / f"ocr_result_dpi_{dpi}.md"
        ocr_out_file.write_text(ocr_result, encoding="utf-8")

    # Xóa file PDF tạm
    if temp_pdf.is_file():
        temp_pdf.unlink()
        
    # 2. Tạo báo cáo Markdown
    report_path = PROJECT_ROOT / "output" / "dpi_benchmark_report.md"
    
    # Tìm mức DPI tối ưu nhất
    # Tiêu chí: Chọn DPI có độ chính xác cao nhất. Nếu độ chính xác bằng nhau hoặc chênh lệch rất nhỏ (< 0.5%),
    # chọn DPI có tổng thời gian chạy thấp nhất (tối ưu hiệu năng).
    best_dpi = dpis[0]
    best_score = results[best_dpi]["similarity"]
    best_time = results[best_dpi]["total_time"]
    
    for dpi in dpis[1:]:
        curr_score = results[dpi]["similarity"]
        curr_time = results[dpi]["total_time"]
        # Nếu độ chính xác cao hơn rõ rệt (> 0.2%)
        if curr_score > best_score + 0.2:
            best_dpi = dpi
            best_score = curr_score
            best_time = curr_time
        # Nếu độ chính xác tương đương nhưng thời gian nhanh hơn nhiều (> 15% nhanh hơn)
        elif abs(curr_score - best_score) <= 0.2 and curr_time < best_time * 0.85:
            best_dpi = dpi
            best_score = curr_score
            best_time = curr_time

    report_content = f"""# Báo cáo đánh giá và tối ưu hóa thông số DPI

Báo cáo này so sánh hiệu năng nhận diện OCR bằng mô hình `qwen3.5:4b` qua các độ phân giải DPI khác nhau để tìm ra cấu hình cân bằng tốt nhất giữa **tốc độ** và **độ chính xác**.

## 📊 Bảng so sánh thông số

| DPI | Thời gian Render (s) | Thời gian OCR (s) | Tổng thời gian (s) | Kích thước ảnh (KB) | Độ chính xác (%) |
|:---:|:--------------------:|:-----------------:|:------------------:|:------------------:|:----------------:|
"""
    for dpi in dpis:
        r = results[dpi]
        report_content += f"| **{dpi}** | {r['time_render']:.3f}s | {r['time_ocr']:.2f}s | {r['total_time']:.2f}s | {r['image_size_kb']:.1f} KB | {r['similarity']:.2f}% |\n"
        
    report_content += f"""
## 💡 Đánh giá chi tiết
*   **DPI thấp (180 - 200)**: Có kích thước ảnh nhỏ, giúp truyền dữ liệu qua API Ollama nhanh hơn và mô hình xử lý tốn ít tài nguyên hơn. Tuy nhiên, nếu chữ quá nhỏ hoặc mờ, độ chính xác có thể bị giảm.
*   **DPI cao (250 - 300)**: Giúp các nét chữ, công thức toán và ký tự nhỏ rõ nét hơn, cải thiện độ chính xác. Nhưng dung lượng ảnh sẽ tăng gấp nhiều lần, khiến thời gian nạp ảnh và thời gian xử lý OCR tăng lên đáng kể.

## 🏆 Đề xuất DPI mặc định tối ưu nhất
Dựa trên kết quả đo lường thực tế: **DPI = {best_dpi}** là cấu hình tối ưu nhất được chọn làm mặc định cho dự án.

*Báo cáo được tạo tự động vào lúc {pymupdf.__write__ if hasattr(pymupdf, '__write__') else ''}*
"""
    report_path.write_text(report_content, encoding="utf-8")
    
    print("=== HOÀN TẤT BENCHMARK ===")
    print(f"Đã lưu báo cáo so sánh chi tiết tại: {report_path.as_posix()}")
    print(f"Đề xuất DPI tối ưu nhất: {best_dpi}")

if __name__ == "__main__":
    run_dpi_benchmark()
