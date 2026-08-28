"""
Module: evaluate_accuracy.py
Nhiệm vụ: Bộ công cụ đo lường độ chính xác định lượng chuẩn xác giữa kết quả OCR và Bản chuẩn (Ground Truth).
Chỉ số đo lường:
  - CER (Character Error Rate) -> Độ chính xác ký tự (%)
  - WER (Word Error Rate) -> Độ chính xác từ ngữ (%)
  - Table F1 -> Độ khớp bảng biểu (%)
  - Formula F1 -> Độ khớp công thức toán (%)
  - Numbers F1 -> Độ chính xác con số (%)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ocr_metrics import score_markdown


def evaluate_folders(output_dir: Path, ground_truth_dir: Path, single_file: str | None = None) -> list[dict]:
    """So sánh toàn bộ file markdown trong thư mục kết quả với thư mục ground truth tương ứng."""
    if not output_dir.exists():
        raise FileNotFoundError(f"Thư mục kết quả OCR không tồn tại: {output_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Thư mục Bản Chuẩn (Ground Truth) không tồn tại: {ground_truth_dir}")

    # Thu thập danh sách file từ 2 thư mục
    output_files = {f.stem.split(".")[0]: f for f in output_dir.glob("*.md")}
    gt_files = {f.stem.split(".")[0]: f for f in ground_truth_dir.glob("*.md")}

    if single_file:
        target_stem = Path(single_file).stem.split(".")[0]
        if target_stem not in output_files:
            raise FileNotFoundError(f"Không tìm thấy file '{target_stem}.md' trong thư mục kết quả '{output_dir.name}/'")
        if target_stem not in gt_files:
            raise FileNotFoundError(f"Không tìm thấy file chuẩn '{target_stem}.md' trong thư mục '{ground_truth_dir.name}/'")
        common_stems = [target_stem]
    else:
        # Tìm các cặp file có tên trùng khớp nhau
        common_stems = sorted(list(set(output_files.keys()) & set(gt_files.keys())))

    print("\n" + "=" * 92)
    print(f" BẢNG ĐÁNH GIÁ ĐỘ CHÍNH XÁC OCR: [{output_dir.name}/] SO VỚI BẢN CHUẨN [{ground_truth_dir.name}/]")
    print("=" * 92)

    if not common_stems:
        print("\n⚠️ KHÔNG TÌM THẤY CẶP FILE NÀO TRÙNG TÊN ĐỂ ĐỐI CHIẾU!")
        print(f"\n📁 Thư mục kết quả OCR [{output_dir.name}/] có các file:")
        out_list = [f.name for f in output_dir.glob('*.md')][:8]
        for name in out_list:
            print(f"   + {name}")
        if len(list(output_dir.glob('*.md'))) > 8:
            print(f"   ... và {len(list(output_dir.glob('*.md'))) - 8} file khác")

        print(f"\n📁 Thư mục Bản Chuẩn [{ground_truth_dir.name}/] có các file:")
        for name in sorted([f.name for f in ground_truth_dir.glob('*.md')]):
            print(f"   + {name}")

        print("\n💡 HƯỚNG DẪN:")
        print("   - Để chấm điểm cho file nào, bạn chỉ cần tạo 1 file chuẩn cùng tên trong thư mục samples/ground_truth/.")
        print("   - Ví dụ: Trong OCR/ có 'A.md' -> tạo thêm 'samples/ground_truth/A.md' rồi chạy lại lệnh.")
        print("=" * 92 + "\n")
        return []

    print(f"{'Tên Tài Liệu':<24} | {'Chính Xác Chữ':<14} | {'Chính Xác Từ':<14} | {'Bảng Biểu':<10} | {'Con Số':<10} | {'Đánh Giá'}")
    print("-" * 92)

    results = []
    for stem in common_stems:
        gt_file = gt_files[stem]
        actual_file = output_files[stem]

        expected_text = gt_file.read_text(encoding="utf-8")
        actual_text = actual_file.read_text(encoding="utf-8")

        metrics = score_markdown(expected_text, actual_text)
        
        char_acc = max(0.0, (1.0 - float(metrics["cer"])) * 100.0)
        word_acc = max(0.0, (1.0 - float(metrics["wer"])) * 100.0)
        table_acc = float(metrics.get("table_f1", 1.0)) * 100.0
        num_acc = float(metrics.get("numbers_f1", 1.0)) * 100.0

        if char_acc >= 98.0 and word_acc >= 95.0:
            rating = "🟢 Xuất sắc"
        elif char_acc >= 90.0:
            rating = "🟡 Đạt chuẩn"
        else:
            rating = "🔴 Cần chú ý"

        print(f"{stem:<24} | {char_acc:>12.2f}% | {word_acc:>12.2f}% | {table_acc:>8.1f}% | {num_acc:>8.1f}% | {rating}")

        results.append({
            "name": stem,
            "char_acc": char_acc,
            "word_acc": word_acc,
            "table_acc": table_acc,
            "num_acc": num_acc,
            "cer": float(metrics["cer"]),
            "wer": float(metrics["wer"]),
            "rating": rating,
            "markdown_valid": metrics["markdown_valid"],
        })

    print("=" * 92)

    if results:
        avg_char = sum(r["char_acc"] for r in results) / len(results)
        avg_word = sum(r["word_acc"] for r in results) / len(results)
        avg_table = sum(r["table_acc"] for r in results) / len(results)
        avg_num = sum(r["num_acc"] for r in results) / len(results)

        print(f"\n📊 TỔNG KẾT TRUNG BÌNH ({len(results)} tài liệu được so khớp):")
        print(f"  - Độ chính xác ký tự (Character Accuracy):  {avg_char:.2f}% (Tỷ lệ sai CER: {100-avg_char:.2f}%)")
        print(f"  - Độ chính xác từ ngữ (Word Accuracy):       {avg_word:.2f}% (Tỷ lệ sai WER: {100-avg_word:.2f}%)")
        print(f"  - Độ chính xác bảng biểu (Table Structure): {avg_table:.2f}%")
        print(f"  - Độ chính xác số liệu (Numbers Accuracy):  {avg_num:.2f}%")
        print("=" * 92 + "\n")

    return results


def export_markdown_report(results: list[dict], output_report_path: Path):
    """Xuất báo cáo chi tiết dạng Markdown."""
    if not results:
        return

    avg_char = sum(r["char_acc"] for r in results) / len(results)
    avg_word = sum(r["word_acc"] for r in results) / len(results)
    avg_table = sum(r["table_acc"] for r in results) / len(results)
    avg_num = sum(r["num_acc"] for r in results) / len(results)

    lines = [
        "# Báo Cáo Đo Lường Độ Chính Xác OCR",
        f"- **Thời điểm đánh giá:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Số lượng tài liệu kiểm thử:** {len(results)}",
        "",
        "## 1. Tổng kết Chỉ số Toàn diện",
        f"- **Độ chính xác ký tự TB (1 - CER):** `{avg_char:.2f}%`",
        f"- **Độ chính xác từ ngữ TB (1 - WER):** `{avg_word:.2f}%`",
        f"- **Độ chính xác bảng biểu TB (Table F1):** `{avg_table:.2f}%`",
        f"- **Độ chính xác con số & thống kê TB:** `{avg_num:.2f}%`",
        "",
        "## 2. Chi tiết từng Tài liệu Kiểm thử",
        "| Tài Liệu | Độ Chính Xác Ký Tự | Độ Chính Xác Từ | Khớp Bảng | Khớp Con Số | Đánh Giá |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lines.append(
            f"| `{r['name']}` | **{r['char_acc']:.2f}%** | {r['word_acc']:.2f}% | {r['table_acc']:.1f}% | {r['num_acc']:.1f}% | {r['rating']} |"
        )

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> Đã xuất báo cáo chi tiết tại: {output_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Bộ công cụ đo lường độ chính xác OCR so với Bản chuẩn (Ground Truth)")
    parser.add_argument("--output", default="OCR", help="Thư mục chứa file markdown kết quả OCR (mặc định: OCR)")
    parser.add_argument("--ground-truth", default="samples/ground_truth", help="Thư mục chứa file Markdown chuẩn mẫu (mặc định: samples/ground_truth)")
    parser.add_argument("--file", default=None, help="Tên file cụ thể cần đánh giá (ví dụ: A hoặc A.md)")
    parser.add_argument("--report", default="benchmark_report.md", help="Đường dẫn lưu file báo cáo markdown")

    args = parser.parse_args()
    
    out_dir = (PROJECT_ROOT / args.output).resolve()
    gt_dir = (PROJECT_ROOT / args.ground_truth).resolve()
    report_file = (PROJECT_ROOT / args.report).resolve()

    try:
        results = evaluate_folders(out_dir, gt_dir, single_file=args.file)
        if results:
            export_markdown_report(results, report_file)
    except Exception as e:
        print(f"Lỗi khi đánh giá: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
