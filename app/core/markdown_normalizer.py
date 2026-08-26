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
    """Normalize local image paths, enclose paths with spaces in <...>, and remove duplicate entries."""
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
        # Enclose target in <...> if it contains whitespace (CommonMark standard)
        formatted_target = f"<{target}>" if " " in target else target
        return f"![{alt.strip()}]({formatted_target})"

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


def strip_assistant_conversational_artifacts(markdown: str) -> str:
    """Remove assistant preambles, prompt leakage, and conversational meta-artifacts."""
    page_blocks = re.split(r"(<!--\s*Page\s+\d+\s*-->)", markdown, flags=re.IGNORECASE)
    cleaned_blocks = []
    from app.core.batch_ocr import is_blank_ocr_response
    for block in page_blocks:
        if re.fullmatch(r"<!--\s*Page\s+\d+\s*-->", block.strip(), re.IGNORECASE):
            cleaned_blocks.append(block)
            continue
        if is_blank_ocr_response(block):
            cleaned_blocks.append("")
            continue
        cleaned = re.sub(
            r"\A\s*(?:(?:Here (?:is|are)|Below is|Dưới đây là|Certainly|Sure|Based on (?:your|the) requirements)[^\n]*?(?:OCR|transcription|extracted|requested|chuyển đổi|văn bản)?[^\n]*?:\s*\n+)+",
            "",
            block,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\n+\s*(?:I hope this (?:helps|is helpful)|Let me know if you (?:need|have)|End of transcription|Đó là toàn bộ nội dung)[^\n]*\Z",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned_blocks.append(cleaned)
    return "".join(cleaned_blocks)


def normalize_checkboxes(markdown: str) -> str:
    """Normalize OCR checkbox artifacts to standard Markdown task list items or form markers."""
    # Convert checked variations like [v], [V], [x], [X], [*], ☑, ✓, ✔
    # 1. Unicode checkboxes at start of line
    markdown = re.sub(r"(?m)^(\s*[-*+]?\s*)[☑☒✓✔]\s*", r"\1- [x] ", markdown)
    markdown = re.sub(r"(?m)^(\s*[-*+]?\s*)☐\s*", r"\1- [ ] ", markdown)

    # 2. [v], [V], [X], [x], [*] at start of line / list
    markdown = re.sub(r"(?m)^(\s*[-*+]?\s*)\[[vVxX*✓✔]\]\s*", r"\1- [x] ", markdown)
    markdown = re.sub(r"(?m)^(\s*[-*+]?\s*)\[\s*\]\s*", r"\1- [ ] ", markdown)

    # 3. Inline checkboxes in form fields (e.g., "Nam [x]  Nữ [ ]")
    markdown = re.sub(r"\[[vVxX*✓✔]\]", "[x]", markdown)
    markdown = re.sub(r"[☑☒]", "[x]", markdown)
    markdown = re.sub(r"☐", "[ ]", markdown)

    # 4. Clean up duplicate bullet markers like "- - [x]" -> "- [x]"
    markdown = re.sub(r"(?m)^\s*[-*+]\s+[-*+]\s+\[([ xX])\]", r"- [\1]", markdown)
    return markdown


def normalize_special_symbols(markdown: str) -> str:
    """Normalize OCR artifacts in units, dimensions, temperatures, and legal/math symbols."""
    # Degrees Celsius: 37 oC, 37 0C, 37oC, 37 °C -> 37°C
    markdown = re.sub(r"(\b\d+(?:[.,]\d+)?)\s*(?:o|O|0|°)\s*C\b", r"\1°C", markdown)
    # Degrees Fahrenheit
    markdown = re.sub(r"(\b\d+(?:[.,]\d+)?)\s*(?:o|O|0|°)\s*F\b", r"\1°F", markdown)
    # Dimension x: 200 x 300 or 200 X 300 -> 200 × 300
    markdown = re.sub(r"(\b\d+(?:[.,]\d+)?)\s*[xX]\s*(\b\d+(?:[.,]\d+)?)", r"\1 × \2", markdown)
    # Micro units: ug, um, ul, us -> µg, µm, µl, µs (in measurement context)
    markdown = re.sub(r"(\b\d+(?:[.,]\d+)?\s*)u([gmlsL])\b", r"\1µ\2", markdown)
    # Plus-minus: + - or +- -> ±
    markdown = re.sub(r"(?<=\d|\s)\+\s*-\s*(?=\d)", "±", markdown)
    # Comparisons: <= -> ≤, >= -> ≥, ~= -> ≈
    markdown = re.sub(r"(?<=\d|\w|\s)<=(?=\s*\d)", "≤", markdown)
    markdown = re.sub(r"(?<=\d|\w|\s)>=(?=\s*\d)", "≥", markdown)
    markdown = re.sub(r"(?<=\d|\w|\s)~=(?=\s*\d)", "≈", markdown)
    # Copyright / Registered / Trademark
    markdown = re.sub(r"\([cC]\)", "©", markdown)
    markdown = re.sub(r"\([rR]\)", "®", markdown)
    markdown = re.sub(r"\((?:tm|TM)\)", "™", markdown)
    return markdown


def collapse_inline_repetitions(markdown: str) -> str:
    """Collapse repeated phrase loops occurring within individual lines."""
    lines = markdown.splitlines()
    output: list[str] = []
    fenced = display_math = False
    html_table_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced and stripped == "$$":
            display_math = not display_math
        html_table_depth += len(re.findall(r"<table\b", stripped, re.IGNORECASE))
        protected = fenced or display_math or html_table_depth > 0
        html_table_depth = max(
            0, html_table_depth - len(re.findall(r"</table>", stripped, re.IGNORECASE))
        )
        if protected or len(line) < 30:
            output.append(line)
            continue

        def clean_line_repetitions(s: str) -> str:
            # Check repeated phrases of length from 8 to 80 chars
            for repeat_len in range(8, min(80, len(s) // 3 + 1)):
                pattern = re.compile(
                    r"((?:,\s*|;\s*|\s+)?([A-Za-zÀ-ỹĐđ0-9\s-]{" + str(repeat_len) + r",}?))(?:\1){2,}",
                    re.IGNORECASE,
                )
                match = pattern.search(s)
                if match:
                    chunk = match.group(1)
                    full_run = match.group(0)
                    s = s.replace(full_run, chunk)
                    # Clean trailing partial fragment if line was truncated
                    phrase_core = match.group(2).strip()
                    words = phrase_core.split()
                    if len(words) >= 2:
                        for partial_count in range(len(words) - 1, 0, -1):
                            partial_prefix = " ".join(words[:partial_count])
                            if s.rstrip().endswith((f", {partial_prefix}", f" {partial_prefix}")):
                                s = re.sub(rf"(?:,\s*|\s+){re.escape(partial_prefix)}\s*$", "", s)
            return s

        prev_line = ""
        current_line = line
        for _ in range(3):
            if current_line == prev_line:
                break
            prev_line = current_line
            current_line = clean_line_repetitions(current_line)

        output.append(current_line)
    return "\n".join(output)


def normalize_structure(
    markdown: str, stats: MarkdownNormalizationStats | None = None
) -> str:
    """Apply safe structural repairs before table/math-specific finalization."""
    markdown = strip_assistant_conversational_artifacts(markdown)
    markdown = collapse_repetition_loops(markdown, stats)
    markdown = collapse_inline_repetitions(markdown)
    markdown = normalize_headings(markdown, stats)
    markdown = normalize_paragraphs_and_lists(markdown, stats)
    markdown = normalize_checkboxes(markdown)
    markdown = normalize_special_symbols(markdown)
    markdown = normalize_images(markdown, stats)
    markdown = remove_standalone_page_artifacts(markdown, stats)
    markdown = stitch_cross_page_paragraphs(markdown, stats)
    markdown = repair_unclosed_table_html(markdown, stats)
    markdown = normalize_html_table_blocks(markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()
