"""
Module: ocr_validator.py
Nhiệm vụ: Kiểm tra và kiểm định tính hợp lệ của tệp Markdown đầu ra sau quá trình OCR.
"""

import sys
import re
from pathlib import Path


# Lớp kiểm định chất lượng tệp OCR Markdown đầu ra
class OCRValidator:

    def __init__(self):
        pass

    # Hàm thực hiện kiểm định một tệp Markdown cụ thể
    def validate_file(self, file_path: Path) -> dict:
        """
        Kiểm định chất lượng của một tệp Markdown:
        - Kiểm tra xem tệp có trống không.
        - Phát hiện lỗi định dạng Markdown (như rào chắn mã chưa đóng, dấu backtick lẻ).
        - Kiểm tra các lỗi toán học LaTeX (thiếu dấu $, không khớp ngoặc nhọn {}).
        - Kiểm tra lỗi bảng biểu (lệch số cột giữa các dòng, không khớp thẻ HTML table).
        - Phát hiện ảo giác của mô hình (lặp lại dòng, lặp từ liên tục).
        """
        content = file_path.read_text(encoding="utf-8").strip()
        results = {
            "file": file_path.name,
            "is_empty": False,
            "markdown_errors": [],
            "formula_errors": [],
            "table_errors": [],
            "hallucinations": [],
        }

        # 1. Kiểm tra xem tệp có trống không
        if not content:
            results["is_empty"] = True
            return results

        # 2. Kiểm tra lỗi định dạng Markdown
        # Khối mã (Code blocks) chưa đóng
        code_blocks = re.findall(r"```", content)
        if len(code_blocks) % 2 != 0:
            results["markdown_errors"].append("Khối mã chưa đóng (số lượng dấu ``` là số lẻ)")

        # Dấu backtick mã dòng chưa đóng
        for line_num, line in enumerate(content.splitlines(), 1):
            inline_backticks = re.findall(r"`", line)
            if len(inline_backticks) % 2 != 0 and "```" not in line:
                results["markdown_errors"].append(f"Dòng {line_num}: Dấu backtick mã dòng chưa đóng")

        # 3. Kiểm tra lỗi công thức toán học
        double_dollars = content.count("$$")
        single_dollars = content.count("$") - (double_dollars * 2)
        if single_dollars % 2 != 0:
            results["formula_errors"].append("Không khớp ký hiệu đô-la đơn ($) cho công thức dòng")
        if double_dollars % 2 != 0:
            results["formula_errors"].append("Không khớp ký hiệu đô-la kép ($$) cho khối công thức")

        # Kiểm tra ngoặc nhọn {} của LaTeX trong công thức
        formulas = re.findall(r"\$(.*?)\$", content)
        for idx, formula in enumerate(formulas):
            open_braces = formula.count("{")
            close_braces = formula.count("}")
            if open_braces != close_braces:
                results["formula_errors"].append(f"Không khớp ngoặc nhọn {{}} trong công thức: {formula[:40]}... (mở {open_braces}, đóng {close_braces})")

        # 4. Kiểm tra lỗi bảng biểu
        # Kiểm tra số lượng cột phân tách bằng dấu | trong bảng Markdown
        lines = content.splitlines()
        in_table = False
        table_cols = 0
        table_start_line = 0
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            is_table_row = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") > 1
            if is_table_row:
                cols = stripped.count("|") - 1
                if not in_table:
                    in_table = True
                    table_cols = cols
                    table_start_line = line_num
                else:
                    if cols != table_cols:
                        # Bỏ qua dòng ngăn cách |---|---|
                        if not re.match(r"^\|[\s\-\:\+\|]*\|$", stripped):
                            results["table_errors"].append(
                                f"Bảng tại dòng {table_start_line}: Dòng {line_num} có {cols} cột, kỳ vọng {table_cols}"
                            )
            else:
                in_table = False

        # Kiểm tra các cặp thẻ HTML bảng
        html_tags = ["table", "tr", "td", "th"]
        for tag in html_tags:
            open_count = len(re.findall(rf"<{tag}\b", content, re.IGNORECASE))
            close_count = len(re.findall(rf"</{tag}>", content, re.IGNORECASE))
            if open_count != close_count:
                results["table_errors"].append(f"Thẻ bảng HTML không khớp: <{tag}> mở {open_count} lần nhưng đóng {close_count} lần")

        # 5. Kiểm tra ảo giác (Lặp dòng hoặc lặp từ liên tục)
        # Phát hiện lặp dòng giống nhau liên tiếp
        dup_count = 0
        last_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == last_line:
                dup_count += 1
            last_line = stripped
        if dup_count > 3:
            results["hallucinations"].append(f"Phát hiện tỷ lệ lặp dòng cao ({dup_count} dòng trùng lặp)")

        # Phát hiện lặp từ liên tục trong một dòng
        for line_num, line in enumerate(lines, 1):
            words = line.split()
            if len(words) > 10:
                consec = 0
                prev_word = ""
                for w in words:
                    if w.lower() == prev_word.lower():
                        consec += 1
                    else:
                        consec = 0
                    if consec > 4:
                        results["hallucinations"].append(f"Dòng {line_num}: Lặp từ liên tiếp ('{w}')")
                        break

        return results


# Hàm chính chạy CLI kiểm định chất lượng Markdown đầu ra
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kiểm định chất lượng file markdown đầu ra từ OCR")
    parser.add_argument("path", help="Đường dẫn đến file markdown hoặc thư mục chứa các file markdown")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Lỗi: Đường dẫn {path} không tồn tại.", file=sys.stderr)
        sys.exit(1)

    files_to_check = []
    if path.is_file():
        files_to_check.append(path)
    else:
        files_to_check.extend(path.glob("**/*.md"))

    if not files_to_check:
        print("Không tìm thấy tệp markdown nào để kiểm định.")
        sys.exit(0)

    validator = OCRValidator()
    print(f"Đang kiểm định {len(files_to_check)} tệp markdown...\n")
    
    any_errors = False
    for file in files_to_check:
        res = validator.validate_file(file)
        print(f"=== File: {res['file']} ===")
        if res["is_empty"]:
            print("  [CẢNH BÁO] Tệp rỗng.")
            any_errors = True
            continue

        errors_found = False
        for category in ["markdown_errors", "formula_errors", "table_errors", "hallucinations"]:
            if res[category]:
                print(f"  [LỖI] {category.replace('_', ' ').capitalize()}:")
                for err in res[category]:
                    print(f"    - {err}")
                errors_found = True
                any_errors = True
        
        if not errors_found:
            print("  [OK] Không phát hiện lỗi.")
            
    if any_errors:
        sys.exit(1)
    else:
        print("\nTất cả các file đều hợp lệ và vượt qua kiểm định!")


if __name__ == "__main__":
    main()

