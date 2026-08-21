"""Deterministic final repairs for OCR-generated Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass


_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")
_LIST_RE = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.*)$")
_HEADING_RE = re.compile(r"^\s*(#{1,})\s*(.*?)\s*#*\s*$")
_BLOCK_START_RE = re.compile(
    r"^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>|```|~~~|<!--|<[/A-Za-z]|\||!\[|\$\$)"
)
_PAGE_FOOTER_RE = re.compile(
    r"^\s*(?:"
    r"(?:trang\s+)?\d+\s*/\s*\d+\s*(?:[-–—|]\s*mã\s+đề(?:\s+thi)?\s+\w+)?"
    r"|\d+\s*\|\s*thông\s+tin\s+tuyển\s+sinh\s+huflit"
    r"|thông\s+tin\s+tuyển\s+sinh\s+huflit(?:\s*\|\s*\d+|\s+\d+)?"
    r")\s*$",
    re.IGNORECASE,
)
_MAGAZINE_FOOTER_RE = re.compile(
    r"^\s*[^\n]*#(?:ttdl|thongtindulich)[\w-]*[^\n]*$",
    re.IGNORECASE,
)
_GENERIC_FOOTER_RE = re.compile(
    r"^\s*(?:"
    r"#\s*(?:[\wÀ-ỹĐđ-]*\d){6,}[\wÀ-ỹĐđ-]*"
    r"|(?:isbn|issn)\s*[:#]?\s*[\dXx-]{8,}"
    r"|(?:mã|số)\s+(?:xuất bản|đăng ký xuất bản|giấy phép)\s*[:#-]?\s*[\w./-]+"
    r"|(?:trang|page)\s+\d{1,4}(?:\s*/\s*\d{1,4})?"
    r")\s*$",
    re.IGNORECASE,
)
_STANDALONE_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:"
    r"[-–—~*•\s]+\d{1,4}[-–—~*•\s]+"
    r"|\[\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*\]"
    r"|(?:\*\*)?[-–—]?\s*\d{1,4}\s*[-–—]?(?:\*\*)?"
    r")\s*$"
)




@dataclass
class MarkdownNormalizationStats:
    headings: int = 0
    paragraph_lines_joined: int = 0
    list_lines_joined: int = 0
    image_paths: int = 0
    duplicate_images: int = 0
    html_tags_closed: int = 0
    page_artifacts_removed: int = 0
    headings_split: int = 0
    repetition_lines_removed: int = 0

    @property
    def total(self) -> int:
        return sum(vars(self).values())


def normalize_headings(markdown: str, stats: MarkdownNormalizationStats | None = None) -> str:
    """Normalize heading syntax and prevent accidental level jumps."""
    split_lines: list[str] = []
    fenced = False
    for source_line in markdown.splitlines():
        stripped = source_line.lstrip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
        # Inline heading splitting is valid only when the physical line itself
        # begins as a heading. This protects prose, code and table cells that
        # legitimately contain Markdown hash characters.
        can_split = not fenced and bool(re.match(r"^#{1,6}\s*\S", stripped))
        parts = re.split(r"\s+(?=#{1,6}\s+\S)", source_line) if can_split else [source_line]
        if can_split and len(parts) == 1:
            side_by_side = re.match(r"^(#{1,6})\s+(.+?)\s+(\*\*([^*]+)\*\*)\s*$", stripped)
            if side_by_side:
                emphasized = side_by_side.group(4).strip()
                letters = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", emphasized)
                if len(letters) >= 8 and letters == letters.upper():
                    marker = side_by_side.group(1)
                    parts = [
                        f"{marker} {side_by_side.group(2).strip()}",
                        f"{marker} {side_by_side.group(3)}",
                    ]
        if stats is not None and len(parts) > 1:
            stats.headings_split += len(parts) - 1
        split_lines.extend(parts)
    markdown = "\n".join(split_lines)
    output: list[str] = []
    previous_level = 0
    fenced = False
    for line in markdown.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        match = None if fenced else _HEADING_RE.match(line)
        if not match or not match.group(2):
            output.append(line)
            continue
        level = min(len(match.group(1)), 6)
        if previous_level and level > previous_level + 1:
            level = previous_level + 1
        previous_level = level
        normalized = f"{'#' * level} {match.group(2).strip()}"
        if stats is not None and normalized != line:
            stats.headings += 1
        output.append(normalized)
    return "\n".join(output)


def normalize_paragraphs_and_lists(markdown: str, stats: MarkdownNormalizationStats | None = None) -> str:
    """Reconnect conservative prose/list continuations broken by OCR lines."""
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    fenced = False
    in_math = False
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            output.append(line)
            index += 1
            continue
        if not fenced and stripped == "$$":
            in_math = not in_math
            output.append(line)
            index += 1
            continue
        if fenced or in_math or not stripped or _BLOCK_START_RE.match(line):
            output.append(line)
            index += 1
            continue

        parts = [stripped]
        while index + 1 < len(lines):
            following = lines[index + 1]
            next_text = following.strip()
            if (
                not next_text
                or _BLOCK_START_RE.match(following)
                or parts[-1].endswith((".", "!", "?", ":", ";", "。", "！", "？"))
            ):
                break
            # Join only a likely sentence continuation, not two independent
            # title-like lines. Indentation is also accepted for list prose.
            if not (next_text[:1].islower() or following[:1].isspace()):
                break
            parts.append(next_text)
            if stats is not None:
                stats.paragraph_lines_joined += 1
            index += 1
        output.append(" ".join(parts))
        index += 1

    # Join indented/lower-case continuation lines into the preceding list item.
    joined: list[str] = []
    for line in output:
        if joined and line.strip() and _LIST_RE.match(joined[-1]):
            if not _BLOCK_START_RE.match(line) and (line[:1].isspace() or line.strip()[:1].islower()):
                joined[-1] = f"{joined[-1].rstrip()} {line.strip()}"
                if stats is not None:
                    stats.list_lines_joined += 1
                continue
        joined.append(line)
    return "\n".join(joined)


def normalize_images(markdown: str, stats: MarkdownNormalizationStats | None = None) -> str:
    """Normalize local image paths and remove exact duplicate image entries."""
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        alt, raw_target = match.groups()
        target = raw_target.strip().strip("<>")
        if not re.match(r"^(?:https?://|data:)", target, re.IGNORECASE):
            target = target.replace("\\", "/")
            target = re.sub(r"^(?:\./)+", "", target)
        if stats is not None and target != raw_target.strip().strip("<>"):
            stats.image_paths += 1
        key = target.casefold()
        if key in seen:
            if stats is not None:
                stats.duplicate_images += 1
            return ""
        seen.add(key)
        return f"![{alt.strip()}]({target})"

    return _IMAGE_RE.sub(replace, markdown)


def repair_unclosed_table_html(markdown: str, stats: MarkdownNormalizationStats | None = None) -> str:
    """Close common table tags left open by model output without inventing cells."""
    lower = markdown.casefold()
    if "<table" not in lower:
        return markdown
    missing: list[str] = []
    for tag in ("td", "th", "tr", "table"):
        opened = len(re.findall(rf"<{tag}\b", lower))
        closed = len(re.findall(rf"</{tag}\s*>", lower))
        missing.extend(f"</{tag}>" for _ in range(max(0, opened - closed)))
    if not missing:
        return markdown
    if stats is not None:
        stats.html_tags_closed += len(missing)
    return f"{markdown.rstrip()}\n" + "".join(missing)


def normalize_html_table_blocks(markdown: str) -> str:
    """Keep raw HTML tables as one Markdown HTML block.

    A blank line terminates a CommonMark HTML block. Vision models sometimes
    insert such a line at a page boundary and indent the following ``tr``/``td``
    tags, which makes the rest of the table render as source code. Remove only
    whitespace-only lines inside complete table blocks and normalize indentation
    of structural table tags; cell text and whitespace outside tables are kept.
    """
    table_re = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
    structural_tag_re = re.compile(
        r"^\s*(</?(?:table|thead|tbody|tfoot|tr|th|td)\b[^>]*>)",
        re.IGNORECASE,
    )

    def normalize(match: re.Match[str]) -> str:
        lines = []
        for line in match.group(0).splitlines():
            if not line.strip():
                continue
            lines.append(structural_tag_re.sub(r"\1", line))
        return "\n".join(lines)

    return table_re.sub(normalize, markdown)


def remove_standalone_page_artifacts(
    markdown: str, stats: MarkdownNormalizationStats | None = None
) -> str:
    """Remove conservative standalone page-number/header/footer patterns."""
    lines = markdown.splitlines()
    kept: list[str] = []
    page_boundaries = [
        index for index, line in enumerate(lines)
        if re.fullmatch(r"\s*<!--\s*Page\s+\d+\s*-->\s*", line, re.IGNORECASE)
    ]
    page_boundaries.append(len(lines))

    def near_page_end(index: int) -> bool:
        boundary = next((value for value in page_boundaries if value > index), len(lines))
        nonblank_after = sum(bool(value.strip()) for value in lines[index + 1:boundary])
        return nonblank_after <= 4

    def near_page_start(index: int) -> bool:
        """True for the first few visible lines after a page marker.

        Rotated scans commonly place the physical footer at the top of the
        corrected image, so its page number is emitted immediately after the
        ``<!-- Page N -->`` boundary rather than at the end of the page block.
        """
        previous_boundary = max(
            (value for value in page_boundaries[:-1] if value < index),
            default=-1,
        )
        nonblank_before = sum(
            bool(value.strip()) for value in lines[previous_boundary + 1:index]
        )
        return previous_boundary >= 0 and nonblank_before <= 2

    for index, line in enumerate(lines):
        is_footer = bool(
            _PAGE_FOOTER_RE.fullmatch(line)
            or _MAGAZINE_FOOTER_RE.fullmatch(line)
            or (_GENERIC_FOOTER_RE.fullmatch(line) and near_page_end(index))
            or (
                _STANDALONE_PAGE_NUMBER_RE.fullmatch(line)
                and (near_page_end(index) or near_page_start(index))
            )
        )
        # Magazine footers are sometimes OCRed as a standalone page number on
        # one line followed by the publication/hashtag on the next line.
        next_nonblank = ""
        for candidate in lines[index + 1:index + 3]:
            if candidate.strip():
                next_nonblank = candidate
                break
        number_before_footer = bool(
            re.fullmatch(r"\s*(?:\*\*)?\d{1,3}(?:\*\*)?\s*", line)
            and _MAGAZINE_FOOTER_RE.fullmatch(next_nonblank)
        )
        if is_footer or number_before_footer:
            if stats is not None:
                stats.page_artifacts_removed += 1
            continue
        kept.append(line)
    return "\n".join(kept)


def _repeatable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _BLOCK_START_RE.match(line):
        return False
    if stripped.startswith(("<", "!", "|")):
        return False
    letters = sum(char.isalpha() for char in stripped)
    return letters >= 15 and len(stripped) >= 24


def collapse_repetition_loops(
    markdown: str, stats: MarkdownNormalizationStats | None = None
) -> str:
    """Collapse high-confidence consecutive repeated text blocks.

    Only exact normalized repetitions of 1–6 prose lines repeated at least
    three times are touched. Fenced code, display math, HTML tables, pipe
    tables, images and other structural blocks remain byte-for-byte intact.
    """
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    fenced = display_math = False
    html_table_depth = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced and stripped == "$$":
            display_math = not display_math
        html_table_depth += len(re.findall(r"<table\b", stripped, re.IGNORECASE))
        protected = fenced or display_math or html_table_depth > 0
        html_table_depth = max(
            0, html_table_depth - len(re.findall(r"</table>", stripped, re.IGNORECASE))
        )
        if protected or not _repeatable_line(lines[index]):
            output.append(lines[index])
            index += 1
            continue

        collapsed = False
        for width in range(1, min(7, (len(lines) - index) // 3 + 1)):

            block = lines[index:index + width]
            if not all(_repeatable_line(line) for line in block):
                continue
            normalized = tuple(re.sub(r"\s+", " ", line.strip()).casefold() for line in block)
            if sum(len(value) for value in normalized) < (40 if width == 1 else 70):
                continue
            repeats = 1
            while index + (repeats + 1) * width <= len(lines):
                candidate = lines[index + repeats * width:index + (repeats + 1) * width]
                candidate_normalized = tuple(
                    re.sub(r"\s+", " ", line.strip()).casefold() for line in candidate
                )
                if candidate_normalized != normalized:
                    break
                repeats += 1
            if repeats < 3:
                continue
            output.extend(block)
            removed = (repeats - 1) * width
            if stats is not None:
                stats.repetition_lines_removed += removed
            index += repeats * width
            collapsed = True
            break
        if not collapsed:
            output.append(lines[index])
            index += 1
    return "\n".join(output)


def stitch_cross_page_paragraphs(
    markdown: str, stats: MarkdownNormalizationStats | None = None
) -> str:
    """Reconnect a sentence/paragraph continuation broken across page boundaries."""
    pattern = re.compile(
        r"([^\n.!?:\;#\->`|*$\s][^\n.!?:\;#`|*$]*?)\n{1,3}\s*(<!--\s*Page\s+\d+\s*-->)\n{1,3}\s*([a-zà-ỹđ\d][^\n]*)",
        re.IGNORECASE | re.UNICODE,
    )

    def repl(m: re.Match[str]) -> str:
        before = m.group(1).rstrip()
        page_tag = m.group(2).strip()
        after = m.group(3).lstrip()
        if after[:1].islower():
            if stats is not None:
                stats.paragraph_lines_joined += 1
            if before.endswith("-"):
                return f"{before[:-1]} {page_tag} {after}"
            return f"{before} {page_tag} {after}"
        return m.group(0)

    return pattern.sub(repl, markdown)


def normalize_structure(
    markdown: str, stats: MarkdownNormalizationStats | None = None
) -> str:
    """Apply safe structural repairs before table/math-specific finalization."""
    markdown = collapse_repetition_loops(markdown, stats)
    markdown = normalize_headings(markdown, stats)
    markdown = normalize_paragraphs_and_lists(markdown, stats)
    markdown = normalize_images(markdown, stats)
    markdown = remove_standalone_page_artifacts(markdown, stats)
    markdown = stitch_cross_page_paragraphs(markdown, stats)
    markdown = repair_unclosed_table_html(markdown, stats)
    markdown = normalize_html_table_blocks(markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()
