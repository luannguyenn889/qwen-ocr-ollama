"""Deterministic quality checks applied before OCR Markdown is persisted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


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
ERROR_WEIGHTS = {
    "empty_page": 100,
    "missing_pages": 100,
    "duplicate_pages": 80,
    "missing_image": 40,
    "unresolved_placeholder": 35,
    "malformed_markdown_table": 25,
    "malformed_html_table": 25,
    "unbalanced_math_delimiters": 15,
    "missing_vietnamese_diacritics": 10,
    "glued_words": 5,
}


@dataclass(frozen=True)
class QualityReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

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
    for paragraph in paragraphs:
        lower = paragraph.casefold()
        words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", lower)
        if len(words) < 12:
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
        if len(words) >= 25:
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
    return high, medium, len(paragraphs)


def _error_weight(error: str) -> int:
    kind = error.partition(":")[0]
    return ERROR_WEIGHTS.get(kind, 20)


def _quality_key(markdown: str, report: QualityReport) -> tuple[int, int, int, int]:
    severe = sum(_error_weight(error) >= 35 for error in report.errors)
    score = sum(_error_weight(error) for error in report.errors)
    warning_score = len(report.warnings)
    content_length = len("".join(markdown.split()))
    return severe, score, warning_score, -content_length


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
    return errors


def evaluate_page(markdown: str, output_dir: str | Path) -> QualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    if not markdown.strip():
        errors.append("empty_page")

    # Link/image targets commonly contain long underscore-separated filenames.
    # Only inspect reader-visible text so asset paths are not mistaken for OCR
    # words that lost their spaces.
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

    if len(re.findall(r"(?<!\\)\$", markdown)) % 2:
        errors.append("unbalanced_math_delimiters")

    errors.extend(_table_errors(markdown))
    if PLACEHOLDER_RE.search(markdown):
        errors.append("unresolved_placeholder")

    root = Path(output_dir)
    for target in IMAGE_RE.findall(markdown):
        target = target.strip()
        if PLACEHOLDER_RE.search(target):
            # Already represented by unresolved_placeholder; avoid reporting
            # the same root cause again as a missing file.
            continue
        if re.match(r"^(?:https?://|data:)", target, re.IGNORECASE):
            continue
        if not (root / target).is_file():
            errors.append(f"missing_image:{target}")

    return QualityReport(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)))


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
    return QualityReport(tuple(errors))
