"""Deterministic quality checks applied before OCR Markdown is persisted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from html.parser import HTMLParser


VIETNAMESE_MARKS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")
PLACEHOLDER_RE = re.compile(r"(?:image|formula)_placeholder", re.IGNORECASE)
VI_FUNCTION_WORDS = {
    "cua", "va", "la", "trong", "duoc", "cho", "voi", "tu", "den",
    "mot", "nhung", "cac", "nay", "theo", "tai", "ve", "khi", "co",
}
VI_COMMON_WORDS = VI_FUNCTION_WORDS | {
    "viet", "nam", "nguoi", "nuoc", "thanh", "pho", "du lich", "phat trien",
    "chuong trinh", "hoat dong", "thong tin", "doanh nghiep", "thi truong",
}
ENGLISH_WORDS = {
    "the", "and", "is", "are", "of", "to", "in", "for", "with", "from",
    "this", "that", "was", "were", "by", "on", "as", "an", "be", "or",
}


def detect_language_profile(markdown: str) -> str:
    """Classify reader-visible prose for prompt/quality routing."""
    text = " ".join(_prose_paragraphs(markdown)).casefold()
    words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", text)
    if not words:
        return "unknown"
    vi_hits = sum(word in VI_FUNCTION_WORDS or any(char in VIETNAMESE_MARKS for char in word) for word in words)
    en_hits = sum(word in ENGLISH_WORDS for word in words)
    if vi_hits >= 3 and en_hits >= 3:
        return "mixed_vi_en"
    if vi_hits >= 3:
        return "vi"
    if en_hits >= 3:
        return "en"
    return "unknown"
PASS_THRESHOLD = 80.0
FATAL_ERROR_TYPES = {
    "empty_page",
    "missing_pages",
    "duplicate_pages",
    "repetition_loop",
    "truncated_page",
}
RETRY_FROM_IMAGE_TYPES = {"missing_vietnamese_diacritics"}

ERROR_WEIGHTS = {
    "empty_page": 100,
    "missing_pages": 100,
    "duplicate_pages": 80,
    "repetition_loop": 50,
    "missing_image": 40,
    "unresolved_placeholder": 35,
    "truncated_page": 35,
    "malformed_markdown_table": 25,
    "malformed_html_table": 25,
    "unclosed_code_blocks": 25,
    "repeated_words": 20,
    "unbalanced_latex_braces": 15,
    "incomplete_latex_command": 15,
    "unbalanced_math_delimiters": 15,
    "answer_label_inside_math": 10,
    "multiple_answers_inside_math": 10,
    "prose_inside_math": 10,
    "non_math_math_block": 10,
    "display_math_inline": 10,
    "adjacent_math_blocks": 10,
    "missing_vietnamese_diacritics": 10,
    "unclosed_inline_code": 10,
    "glued_words": 5,
}


def _is_fatal_error(error: str) -> bool:
    kind = error.partition(":")[0]
    return kind in FATAL_ERROR_TYPES


@dataclass(frozen=True)
class QualityReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    score: float = 100.0
    fatal_errors: tuple[str, ...] = ()
    should_retry: bool = False

    def __post_init__(self):
        if not self.fatal_errors and self.errors:
            fatals = tuple(e for e in self.errors if _is_fatal_error(e))
            object.__setattr__(self, "fatal_errors", fatals)
        if self.score == 100.0 and self.errors:
            penalties = sum(_error_weight(e) for e in self.errors)
            calculated_score = max(0.0, 100.0 - penalties)
            object.__setattr__(self, "score", calculated_score)
        if not self.should_retry and (
            self.fatal_errors or self.score < PASS_THRESHOLD
            or any(_error_weight(e) >= 30 for e in self.errors)
            or any(e.partition(":")[0] in RETRY_FROM_IMAGE_TYPES for e in self.errors)
        ):
            object.__setattr__(self, "should_retry", True)

    @property
    def passed(self) -> bool:
        return not self.errors


def _prose_paragraphs(markdown: str) -> list[str]:
    """Return prose suitable for language checks, excluding technical content."""
    text = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", "", text, flags=re.DOTALL)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = re.sub(r"https?://\S+|\b\S+@\S+\.\S+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<table\b.*?</table>", "", text, flags=re.IGNORECASE | re.DOTALL)

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("|", "<!--")):
                continue
            plain = re.sub(r"^#{1,6}\s+", "", stripped)
            letters = [char for char in plain if char.isalpha()]
            if len(letters) < 60 or len(plain.split()) < 12:
                continue
            if letters and plain.isupper():
                continue
            if sum(char.isdigit() for char in plain) / max(len(plain), 1) > 0.20:
                continue
            lines.append(plain)
        paragraph = " ".join(lines).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def _vietnamese_diacritic_confidence(markdown: str) -> tuple[int, int, int]:
    """Return counts of high- and medium-confidence paragraphs missing accents."""
    high = medium = 0
    paragraphs = _prose_paragraphs(markdown)
    # Score sentence-sized segments. A whole mixed-language paragraph can have
    # enough English function words to hide an adjacent Vietnamese passage that
    # lost its accents.
    segments = [
        segment.strip()
        for paragraph in paragraphs
        for segment in re.split(r"(?<=[.!?;:])\s+|\n+", paragraph)
        if segment.strip()
    ]
    for paragraph in segments:
        lower = paragraph.casefold()
        words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", lower)
        if len(words) < 8:
            continue
        word_set = set(words)
        function_hits = sum(word in VI_FUNCTION_WORDS for word in words)
        phrase_hits = sum(lower.count(term) for term in VI_COMMON_WORDS)
        english_hits = sum(word in ENGLISH_WORDS for word in words)
        letters = [char for char in lower if char.isalpha()]
        mark_ratio = sum(char in VIETNAMESE_MARKS for char in letters) / max(len(letters), 1)
        technical_ratio = sum(bool(re.search(r"\d|[_/@]", token)) for token in paragraph.split()) / max(len(paragraph.split()), 1)

        score = 0
        if function_hits >= 3:
            score += 3
        if phrase_hits >= 5 or len(word_set & VI_FUNCTION_WORDS) >= 4:
            score += 2
        if len(words) >= 18:
            score += 1
        if mark_ratio < 0.02:
            score += 2
        if technical_ratio >= 0.15:
            score -= 3
        if english_hits >= 4 and english_hits > function_hits:
            score -= 4

        if score >= 7:
            high += 1
        elif score >= 6:
            medium += 1
    return high, medium, len(segments)


class _TableStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.rows: list[list[tuple[int, int]]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_map = dict(attrs)
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.rows.append([])
        elif tag in {"td", "th"} and self.in_row:
            try:
                colspan = max(1, int(attrs_map.get("colspan", "1")))
                rowspan = max(1, int(attrs_map.get("rowspan", "1")))
            except (TypeError, ValueError):
                colspan = rowspan = 0
            self.rows[-1].append((colspan, rowspan))

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr":
            self.in_row = False
        elif tag == "table":
            self.in_table = False


def html_table_structure_errors(markdown: str) -> list[str]:
    """Validate logical row widths, including cells carried by rowspan."""
    errors: list[str] = []
    for table in re.findall(r"<table\b.*?</table>", markdown, re.IGNORECASE | re.DOTALL):
        # A non-greedy regex fragment ends at the inner closing tag for nested
        # tables. Leave those to the tag-balance validator instead of emitting
        # a false logical-column error for the outer table.
        if len(re.findall(r"<table\b", table, re.IGNORECASE)) != 1:
            continue
        parser = _TableStructureParser()
        parser.feed(table)
        active: dict[int, int] = {}
        widths: list[int] = []
        for cells in parser.rows:
            occupied = {column for column, remaining in active.items() if remaining > 0}
            cursor = 0
            for colspan, rowspan in cells:
                if colspan < 1 or rowspan < 1:
                    errors.append("malformed_html_table:invalid_span")
                    continue
                while cursor in occupied:
                    cursor += 1
                for column in range(cursor, cursor + colspan):
                    if column in occupied:
                        errors.append("malformed_html_table:overlapping_span")
                    if rowspan > 1:
                        active[column] = max(active.get(column, 0), rowspan)
                cursor += colspan
            width = max([cursor, *(column + 1 for column in occupied)] or [0])
            widths.append(width)
            active = {column: remaining - 1 for column, remaining in active.items() if remaining > 1}
        nonempty = [width for width in widths if width]
        if nonempty and len(set(nonempty)) > 1:
            errors.append("malformed_html_table:inconsistent_logical_columns")
    return list(dict.fromkeys(errors))


def _error_weight(error: str) -> int:
    kind = error.partition(":")[0]
    return ERROR_WEIGHTS.get(kind, 20)


def _quality_key(markdown: str, report: QualityReport) -> tuple[int, int, int, int, int]:
    fatal_count = len(report.fatal_errors)
    severe = sum(_error_weight(error) >= 35 for error in report.errors)
    penalty = int(100.0 - report.score)
    warning_score = len(report.warnings)
    content_length = len("".join(markdown.split()))
    return fatal_count, severe, penalty, warning_score, -content_length


def choose_best_page(
    first_markdown: str,
    first_report: QualityReport,
    retry_markdown: str,
    retry_report: QualityReport,
) -> tuple[str, QualityReport]:
    """Return the strongest available OCR result after a quality retry.

    Severe failures and weighted error cost take precedence over raw error
    count. Warnings and content length are deterministic tie-breakers.
    """
    first_key = _quality_key(first_markdown, first_report)
    retry_key = _quality_key(retry_markdown, retry_report)
    if retry_key < first_key:
        return retry_markdown, retry_report
    return first_markdown, first_report


def _detect_markdown_syntax_errors(markdown: str) -> list[str]:
    """Kiểm tra tính toàn vẹn cú pháp Markdown cơ bản."""
    errors: list[str] = []
    # Kiểm tra khối mã ``` chưa đóng
    code_fences = re.findall(r"```", markdown)
    if len(code_fences) % 2 != 0:
        errors.append("unclosed_code_blocks")

    # Kiểm tra backtick inline lẻ trên các dòng không phải khối mã
    for line in markdown.splitlines():
        if "```" in line:
            continue
        inline_ticks = re.findall(r"(?<!`)`(?!`)", line)
        if len(inline_ticks) % 2 != 0:
            errors.append("unclosed_inline_code")
            break
    return errors


def _detect_hallucination_and_repetition(markdown: str) -> list[str]:
    """Phát hiện lỗi lặp lại dòng, lặp từ hoặc vòng lặp sinh token của AI."""
    errors: list[str] = []
    lines = [line.strip() for line in markdown.splitlines() if line.strip() and not line.strip().startswith("<!--")]

    # Use the same conservative block detector as finalization so multi-line
    # loops (A-B-A-B-A-B) trigger a page retry before persistence.
    from app.core.markdown_normalizer import MarkdownNormalizationStats, collapse_repetition_loops
    repetition_stats = MarkdownNormalizationStats()
    collapse_repetition_loops(markdown, repetition_stats)
    if repetition_stats.repetition_lines_removed:
        errors.append("repetition_loop")

    # 1. Phát hiện lặp dòng liên tiếp (>= 3 dòng giống hệt nhau)
    dup_lines = 0
    last_line = ""
    for line in lines:
        if len(line) >= 10 and line == last_line:
            dup_lines += 1
        last_line = line
    if dup_lines >= 3:
        errors.append("repetition_loop")

    # 2. Phát hiện lặp từ liên tiếp trong 1 dòng (> 4 từ giống nhau)
    for line in lines:
        words = re.findall(r"[a-zA-ZÀ-ỹĐđ0-9]+", line)
        if len(words) >= 8:
            consec = 0
            prev_w = ""
            for w in words:
                if w.casefold() == prev_w.casefold():
                    consec += 1
                    if consec >= 4:
                        errors.append("repeated_words")
                        break
                else:
                    consec = 0
                prev_w = w
    return list(dict.fromkeys(errors))


def _table_errors(markdown: str) -> list[str]:
    errors: list[str] = []
    rows = []
    for line_number, line in enumerate(markdown.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            columns = len(stripped.split("|")) - 2
            rows.append((line_number, columns))
        elif rows:
            expected = rows[0][1]
            if any(columns != expected for _, columns in rows):
                errors.append(f"malformed_markdown_table:{rows[0][0]}")
            rows = []
    if rows:
        expected = rows[0][1]
        if any(columns != expected for _, columns in rows):
            errors.append(f"malformed_markdown_table:{rows[0][0]}")
    lower = markdown.casefold()
    for tag in ("table", "tr", "td", "th"):
        if len(re.findall(rf"<{tag}\b", lower)) != len(re.findall(rf"</{tag}>", lower)):
            errors.append(f"malformed_html_table:{tag}")
    errors.extend(html_table_structure_errors(markdown))
    return errors


def evaluate_page(
    markdown: str, output_dir: str | Path, *, check_tables: bool = True
) -> QualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    if not markdown.strip():
        errors.append("empty_page")

    # 1. Cú pháp Markdown & Bảng
    errors.extend(_detect_markdown_syntax_errors(markdown))
    if check_tables:
        errors.extend(_table_errors(markdown))

    # 2. Phát hiện lặp từ / ảo giác (Hallucinations)
    errors.extend(_detect_hallucination_and_repetition(markdown))

    # 3. Ký tự dính / Dấu tiếng Việt
    visible_text = MARKDOWN_LINK_RE.sub(r"\1", markdown)
    visible_text = re.sub(r"https?://\S+", "", visible_text, flags=re.IGNORECASE)
    glued_candidates = re.findall(r"(?<![\w_-])[^\W\d_]{45,}(?![\w_-])", visible_text, re.UNICODE)
    glued_signals = ("cua", "va", "la", "trong", "duoc", "nhung", "mot", "cho", "voi", "viet", "nam")
    if any(
        token == token.casefold()
        and (len(token) >= 80 or sum(signal in token.casefold() for signal in glued_signals) >= 3)
        for token in glued_candidates
    ):
        errors.append("glued_words")

    high_confidence, medium_confidence, prose_count = _vietnamese_diacritic_confidence(markdown)
    if high_confidence:
        errors.append("missing_vietnamese_diacritics")
    elif medium_confidence >= 2 and medium_confidence / max(prose_count, 1) >= 0.25:
        warnings.append("suspected_missing_vietnamese_diacritics")

    # 4. Kiểm tra tính hợp lệ của LaTeX
    if len(re.findall(r"(?<!\\)\$", markdown)) % 2:
        errors.append("unbalanced_math_delimiters")
    from app.core.math_cleanup import math_quality_errors
    errors.extend(math_quality_errors(markdown))

    # 5. Hình ảnh và placeholder
    if PLACEHOLDER_RE.search(markdown):
        errors.append("unresolved_placeholder")

    root = Path(output_dir)
    for target in IMAGE_RE.findall(markdown):
        target = target.strip()
        if PLACEHOLDER_RE.search(target):
            continue
        if re.match(r"^(?:https?://|data:)", target, re.IGNORECASE):
            continue
        if not (root / target).is_file():
            errors.append(f"missing_image:{target}")

    unique_errors = tuple(dict.fromkeys(errors))
    unique_warnings = tuple(dict.fromkeys(warnings))

    # Tính điểm chất lượng (0 - 100)
    penalties = sum(_error_weight(e) for e in unique_errors)
    score = max(0.0, round(100.0 - penalties, 1))
    fatals = tuple(e for e in unique_errors if _is_fatal_error(e))
    should_retry = (
        bool(fatals) or score < PASS_THRESHOLD
        or any(_error_weight(e) >= 30 for e in unique_errors)
        or any(e.partition(":")[0] in RETRY_FROM_IMAGE_TYPES for e in unique_errors)
    )

    return QualityReport(
        errors=unique_errors,
        warnings=unique_warnings,
        score=score,
        fatal_errors=fatals,
        should_retry=should_retry,
    )


def validate_page_numbers(page_numbers: list[int], total_pages: int) -> QualityReport:
    expected = set(range(1, total_pages + 1))
    actual = set(page_numbers)
    errors = []
    missing = sorted(expected - actual)
    duplicates = sorted(number for number in actual if page_numbers.count(number) > 1)
    if missing:
        errors.append("missing_pages:" + ",".join(map(str, missing)))
    if duplicates:
        errors.append("duplicate_pages:" + ",".join(map(str, duplicates)))

    unique_errors = tuple(errors)
    fatals = tuple(e for e in unique_errors if _is_fatal_error(e))
    penalties = sum(_error_weight(e) for e in unique_errors)
    score = max(0.0, round(100.0 - penalties, 1))

    return QualityReport(
        errors=unique_errors,
        score=score,
        fatal_errors=fatals,
        should_retry=bool(fatals) or score < PASS_THRESHOLD,
    )
