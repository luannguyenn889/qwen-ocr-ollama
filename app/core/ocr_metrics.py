"""
Module: ocr_metrics.py
Nhiệm vụ: Tính toán các chỉ số chất lượng định lượng cho kết quả OCR so với nhãn chuẩn (Benchmark).
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


# Hàm chuẩn hóa văn bản
def normalize_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng của văn bản; giữ nguyên dấu câu và kiểu chữ."""
    return " ".join(text.casefold().split())


# Hàm tính khoảng cách Levenshtein giữa hai danh sách phần tử
def _distance(expected: list[str], actual: list[str]) -> int:
    """Tính toán khoảng cách Levenshtein (edit distance) giữa hai danh sách."""
    previous = list(range(len(actual) + 1))
    for index, item in enumerate(expected, start=1):
        current = [index]
        for actual_index, other in enumerate(actual, start=1):
            current.append(min(
                previous[actual_index] + 1,
                current[actual_index - 1] + 1,
                previous[actual_index - 1] + (item != other),
            ))
        previous = current
    return previous[-1]


# Hàm tính tỷ lệ lỗi chung giữa hai tập hợp phần tử
def error_rate(expected: Iterable[str], actual: Iterable[str]) -> float:
    """Tính tỷ lệ lỗi dựa trên khoảng cách Levenshtein chia cho tổng số phần tử kỳ vọng."""
    expected_items, actual_items = list(expected), list(actual)
    if not expected_items:
        return 0.0 if not actual_items else 1.0
    return _distance(expected_items, actual_items) / len(expected_items)


# Hàm tính tỷ lệ lỗi ký tự (Character Error Rate)
def cer(expected: str, actual: str) -> float:
    """Tính tỷ lệ lỗi ký tự (CER) giữa chuỗi kỳ vọng và chuỗi thực tế."""
    return error_rate(list(normalize_text(expected)), list(normalize_text(actual)))


# Hàm tính tỷ lệ lỗi từ (Word Error Rate)
def wer(expected: str, actual: str) -> float:
    """Tính tỷ lệ lỗi từ (WER) giữa chuỗi kỳ vọng và chuỗi thực tế."""
    return error_rate(normalize_text(expected).split(), normalize_text(actual).split())


# Hàm tính điểm F1 cho hai bộ đếm phần tử
def _f1(expected: Counter[str], actual: Counter[str]) -> float:
    """Tính toán chỉ số F-score (F1) dựa trên sự khớp nhau của bộ đếm."""
    matched = sum((expected & actual).values())
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    precision = matched / sum(actual.values())
    recall = matched / sum(expected.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# Biểu thức chính quy cho biểu thức toán học và số
MATH_RE = re.compile(r"\$\$(.+?)\$$|\$(.+?)\$", re.DOTALL)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?:%|\b)")


# Trích xuất các khối toán học thành bộ đếm phần tử
def _math_items(text: str) -> Counter[str]:
    """Trích xuất các khối công thức toán học LaTeX thành Counter các phần tử chuẩn hóa."""
    return Counter(" ".join((block or inline).split()) for block, inline in MATH_RE.findall(text))


# Trích xuất các số thành bộ đếm phần tử
def _number_items(text: str) -> Counter[str]:
    """Trích xuất tất cả các số trong văn bản thành Counter các phần tử."""
    return Counter(NUMBER_RE.findall(text))


# Hàm đối sánh điểm F1 chính xác của số và công thức toán học
def exact_items(expected: str, actual: str) -> dict[str, float | bool]:
    """Tính toán điểm F1 và độ khớp tuyệt đối của các con số và công thức LaTeX."""
    expected_numbers, actual_numbers = _number_items(expected), _number_items(actual)
    expected_math, actual_math = _math_items(expected), _math_items(actual)
    return {
        "numbers_f1": _f1(expected_numbers, actual_numbers),
        "numbers_exact": expected_numbers == actual_numbers,
        "formulas_f1": _f1(expected_math, actual_math),
        "formulas_exact": expected_math == actual_math,
    }


# Biểu thức chính quy phát hiện tiêu đề danh sách
LIST_RE = re.compile(r"^\s*(?:[-*+] |\d+[.)] )")


# Hàm đếm số bảng biểu và danh sách trong markdown
def _structures(markdown: str) -> tuple[Counter[str], Counter[str]]:
    """Phân tích cấu trúc bảng biểu và danh sách trong văn bản Markdown."""
    tables: Counter[str] = Counter()
    lists: Counter[str] = Counter()
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                tables[f"cells:{len(cells)}"] += 1
        if LIST_RE.match(stripped):
            kind = "ordered" if re.match(r"^\d+[.)] ", stripped) else "unordered"
            lists[kind] += 1
    return tables, lists


# Hàm đối sánh điểm F1 cho cấu trúc tài liệu
def structure_f1(expected: str, actual: str) -> dict[str, float]:
    """Tính toán điểm F1 cho cấu trúc bảng và danh sách Markdown."""
    expected_tables, expected_lists = _structures(expected)
    actual_tables, actual_lists = _structures(actual)
    return {
        "table_f1": _f1(expected_tables, actual_tables),
        "list_f1": _f1(expected_lists, actual_lists),
    }


# Hàm kiểm tra định dạng lỗi cú pháp Markdown được tạo ra
def markdown_validation_errors(markdown: str) -> list[str]:
    """Kiểm tra các lỗi cú pháp Markdown phổ biến (rào chắn mã, đô-la toán học chưa đóng)."""
    errors: list[str] = []
    if markdown.count("```") % 2:
        errors.append("unclosed_code_fence")
    if len(re.findall(r"(?<!\\)\$", markdown)) % 2:
        errors.append("unbalanced_inline_math")
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        if line.strip().count("|") == 1:
            errors.append(f"malformed_table_line:{line_number}")
    return errors


# Hàm đánh giá toàn diện tệp Markdown
def score_markdown(expected: str, actual: str) -> dict[str, float | bool | str]:
    """Đánh giá toàn diện chất lượng tệp Markdown bao gồm CER, WER, độ hợp lệ Markdown, công thức và cấu trúc."""
    result: dict[str, float | bool | str] = {
        "cer": cer(expected, actual),
        "wer": wer(expected, actual),
        "markdown_valid": not markdown_validation_errors(actual),
        "markdown_errors": ";".join(markdown_validation_errors(actual)),
    }
    result.update(exact_items(expected, actual))
    result.update(structure_f1(expected, actual))
    return result

