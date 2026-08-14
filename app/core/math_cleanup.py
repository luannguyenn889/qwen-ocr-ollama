"""Shared deterministic cleanup and validation for OCR mathematics."""

from __future__ import annotations

import re

ANSWER_LABEL_RE = re.compile(r"\s*([A-D])\.\s+")


def normalize_answer_math(text: str) -> str:
    """Keep multiple-choice labels outside math delimiters."""
    def split_inline(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = re.split(r"(\s*[A-D]\.\s+)", inner)
        if len(parts) == 1:
            return match.group(0)
        output: list[str] = []
        for part in parts:
            if re.fullmatch(r"\s*[A-D]\.\s+", part):
                output.append(part)
            elif part.strip():
                output.append(f"${part.strip()}$")
        return "".join(output)

    text = re.sub(r"\$\$\s*([A-D])\.\s*\$\$", r"\1.", text)
    text = re.sub(r"\$\s*([A-D])\.\s*\$", r"\1.", text)
    text = re.sub(r"\$([^$\n]+)\$", split_inline, text)
    text = re.sub(r"(?<=\$)\s*(?=[A-D]\.\s)", " ", text)
    return text


def math_quality_errors(text: str) -> list[str]:
    errors: list[str] = []
    if re.search(r"\${1,2}\s*[A-D]\.\s*\${1,2}", text):
        errors.append("answer_label_inside_math")
    for inner in re.findall(r"(?<!\$)\$([^$\n]+)\$(?!\$)", text):
        if len(re.findall(r"(?:^|\s)[A-D]\.\s", inner)) >= 2:
            errors.append("multiple_answers_inside_math")
        words = re.findall(r"[A-Za-zÀ-ỹĐđ]{3,}", inner)
        operators = len(re.findall(r"[=+*/^_{}\\<>]", inner))
        if len(words) >= 6 and operators <= 1:
            errors.append("prose_inside_math")
        if re.fullmatch(r"\s*(?:[A-D]\.|[.,;:!?-]+)\s*", inner):
            errors.append("non_math_math_block")
    if re.search(r"\$\$[^$\n]+\$\$\S", text) or re.search(r"\S\$\$[^$\n]+\$\$", text):
        errors.append("display_math_inline")
    if re.search(r"\${1,2}\s*\${1,2}", text):
        errors.append("adjacent_math_blocks")
    return list(dict.fromkeys(errors))
