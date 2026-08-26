"""Shared deterministic cleanup and validation for OCR mathematics."""

from __future__ import annotations

import re

ANSWER_LABEL_RE = re.compile(r"\s*([A-D])\.\s+")


def _split_choice_sequence(line: str) -> str:
    """Split only a credible sequence of two or more answer labels."""
    label_re = re.compile(r"(?<!\w)([A-Da-d])[.)]\s*")
    matches = [
        match for match in label_re.finditer(line)
        if len(re.findall(r"(?<!\\)\$", line[:match.start()])) % 2 == 0
    ]
    math_before_single_choice = (
        len(matches) == 1
        and matches[0].group(1).casefold() != "a"
        and "$" in line[:matches[0].start()]
    )
    if len(matches) < 2 and not math_before_single_choice:
        return line
    ranks = ["abcd".index(match.group(1).casefold()) for match in matches]
    if any(right <= left for left, right in zip(ranks, ranks[1:])):
        return line
    is_pipe_table = line.strip().startswith("|") and line.strip().endswith("|")
    is_html_cell = bool(re.search(r"<(?:td|th)\b", line, re.IGNORECASE))
    separator = "<br>" if is_pipe_table or is_html_cell else "\n"
    first_start = matches[0].start()
    starts_with_choice = not line[:first_start].strip()
    split_from = 1 if starts_with_choice else 0
    pieces: list[str] = []
    cursor = 0
    for match_index, match in enumerate(matches):
        if match_index < split_from:
            continue
        pieces.append(line[cursor:match.start()].rstrip())
        pieces.append(separator)
        cursor = match.start()
    pieces.append(line[cursor:])
    return "".join(pieces).strip()


def normalize_answer_math(text: str) -> str:
    """Keep multiple-choice labels outside math delimiters."""
    text = re.sub(r"\\+hfill\b\s*", " ", text)
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

    text = re.sub(r"(?<!\S)\$\$\s*([A-D])\.\s*\$\$(?!\S)", r"\1.", text)
    text = re.sub(r"(?<!\S)\$\s*([A-D])\.\s*\$(?!\S)", r"\1.", text)
    text = re.sub(r"\$([^$\n]+)\$", split_inline, text)
    text = re.sub(r"(?<=\$)[ \t]*(?=[A-D]\.\s)", " ", text)
    normalized_lines = [_split_choice_sequence(line) for line in text.splitlines()]
    text = "\n".join(normalized_lines)
    text = re.sub(r"\n[ \t]*\n(?=[B-D]\.\s*)", "\n", text)
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

        # Kiểm tra cân bằng ngoặc nhọn trong công thức LaTeX
        if inner.count("{") != inner.count("}"):
            errors.append("unbalanced_latex_braces")

        # Kiểm tra các lệnh phân số \frac hoặc căn thức \sqrt chưa hoàn thành
        if re.search(r"\\(?:frac|sqrt)\s*$", inner) or re.search(r"\\frac\s*\{[^{}]*\}\s*$", inner):
            errors.append("incomplete_latex_command")

    if re.search(r"\$\$[^$\n]+\$\$\S", text) or re.search(r"\S\$\$[^$\n]+\$\$", text):
        errors.append("display_math_inline")
    if re.search(r"\${1,2}\s*\${1,2}", text):
        errors.append("adjacent_math_blocks")
    return list(dict.fromkeys(errors))

