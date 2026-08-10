"""Deterministic quality metrics for OCR-to-Markdown benchmark runs."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


def normalize_text(text: str) -> str:
    """Normalise only presentation whitespace; do not discard punctuation or case."""
    return " ".join(text.casefold().split())


def _distance(expected: list[str], actual: list[str]) -> int:
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


def error_rate(expected: Iterable[str], actual: Iterable[str]) -> float:
    expected_items, actual_items = list(expected), list(actual)
    if not expected_items:
        return 0.0 if not actual_items else 1.0
    return _distance(expected_items, actual_items) / len(expected_items)


def cer(expected: str, actual: str) -> float:
    return error_rate(list(normalize_text(expected)), list(normalize_text(actual)))


def wer(expected: str, actual: str) -> float:
    return error_rate(normalize_text(expected).split(), normalize_text(actual).split())


def _f1(expected: Counter[str], actual: Counter[str]) -> float:
    matched = sum((expected & actual).values())
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    precision = matched / sum(actual.values())
    recall = matched / sum(expected.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?:%|\b)")


def _math_items(text: str) -> Counter[str]:
    return Counter(" ".join((block or inline).split()) for block, inline in MATH_RE.findall(text))


def _number_items(text: str) -> Counter[str]:
    return Counter(NUMBER_RE.findall(text))


def exact_items(expected: str, actual: str) -> dict[str, float | bool]:
    """Exact F1 and whole-set match for values and LaTex formulas."""
    expected_numbers, actual_numbers = _number_items(expected), _number_items(actual)
    expected_math, actual_math = _math_items(expected), _math_items(actual)
    return {
        "numbers_f1": _f1(expected_numbers, actual_numbers),
        "numbers_exact": expected_numbers == actual_numbers,
        "formulas_f1": _f1(expected_math, actual_math),
        "formulas_exact": expected_math == actual_math,
    }


LIST_RE = re.compile(r"^\s*(?:[-*+] |\d+[.)] )")


def _structures(markdown: str) -> tuple[Counter[str], Counter[str]]:
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


def structure_f1(expected: str, actual: str) -> dict[str, float]:
    expected_tables, expected_lists = _structures(expected)
    actual_tables, actual_lists = _structures(actual)
    return {
        "table_f1": _f1(expected_tables, actual_tables),
        "list_f1": _f1(expected_lists, actual_lists),
    }


def markdown_validation_errors(markdown: str) -> list[str]:
    """Small, dependency-free validation for generated Markdown conventions."""
    errors: list[str] = []
    if markdown.count("```") % 2:
        errors.append("unclosed_code_fence")
    if len(re.findall(r"(?<!\\)\$", markdown)) % 2:
        errors.append("unbalanced_inline_math")
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        if line.strip().count("|") == 1:
            errors.append(f"malformed_table_line:{line_number}")
    return errors


def score_markdown(expected: str, actual: str) -> dict[str, float | bool | str]:
    result: dict[str, float | bool | str] = {
        "cer": cer(expected, actual),
        "wer": wer(expected, actual),
        "markdown_valid": not markdown_validation_errors(actual),
        "markdown_errors": ";".join(markdown_validation_errors(actual)),
    }
    result.update(exact_items(expected, actual))
    result.update(structure_f1(expected, actual))
    return result
