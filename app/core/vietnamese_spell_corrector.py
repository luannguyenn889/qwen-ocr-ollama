"""Conservative, context-only Vietnamese diacritic correction for OCR output."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from typing import TypedDict


LEXICON_PATH = Path(__file__).with_name("vietnamese_lexicon.json")
WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+", re.UNICODE)
MASK_RE = re.compile(
    r"```.*?```"                         # fenced code
    r"|`[^`\n]*`"                        # inline code
    r"|\$\$.*?\$\$"                    # display math
    r"|(?<!\$)\$[^$\n]*\$(?!\$)"       # inline math
    r"|!?\[[^\]]*\]\([^)]+\)"           # Markdown image/link
    r"|https?://[^\s<>]+"                # URLs
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"  # email
    r"|\b\d{1,4}/\d{4}/[A-ZĐ-]+\b"       # legal document numbers
    r"|\b(?=[A-Za-z0-9._/-]*\d)(?=[A-Za-z0-9._/-]*[-_/])[A-Za-z0-9]+(?:[-_/][A-Za-z0-9.]+)+\b",
    re.IGNORECASE | re.DOTALL,
)
TABLE_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)


class SpellingWarning(TypedDict):
    word: str
    line: int
    suggestion: str
    confidence: str


def _unaccented(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )


@lru_cache(maxsize=1)
def _load_lexicon() -> tuple[
    frozenset[str],
    dict[str, tuple[str, ...]],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str, str]],
]:
    """Load the replaceable JSON lexicon once per process."""
    try:
        data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return frozenset(), {}, frozenset(), frozenset()

    words = frozenset(str(word).casefold() for word in data.get("words", []))
    candidates: dict[str, tuple[str, ...]] = {}
    for plain, accented in data.get("accent_candidates", {}).items():
        values = tuple(dict.fromkeys(str(word).casefold() for word in accented))
        if values:
            candidates[_unaccented(str(plain))] = values
    bigrams = frozenset(
        (parts[0].casefold(), parts[1].casefold())
        for item in data.get("bigrams", [])
        if len(parts := str(item).split()) == 2
    )
    trigrams = frozenset(
        (parts[0].casefold(), parts[1].casefold(), parts[2].casefold())
        for item in data.get("trigrams", [])
        if len(parts := str(item).split()) == 3
    )
    return words, candidates, bigrams, trigrams


def _case_style(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:].lower()
    return replacement.lower() if source.islower() else replacement


def _options(token: str, words: frozenset[str], candidates: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    normalized = token.casefold()
    unacc = _unaccented(normalized)
    pool = candidates.get(unacc, ())
    if not pool:
        return (normalized,) if normalized in words else ()
    res = [normalized] if (normalized in pool or normalized in words) else []
    for item in pool:
        if item not in res:
            res.append(item)
    return tuple(res)


def _has_vn_diacritic(value: str) -> bool:
    return any(
        unicodedata.category(c) == "Mn" or c.casefold() in ("đ", "ă", "â", "ê", "ô", "ơ", "ư")
        for c in value
    )


REPEATED_WORDS_RE = re.compile(
    r"\b(các|những|và|là|của|trong|được|có|với|đã|đang|sẽ|cho|về|tại|từ|bởi|do)\s+\1\b",
    re.IGNORECASE | re.UNICODE,
)


def _dedup_repeated_words(text: str) -> tuple[str, int]:
    count = 0
    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        return m.group(1)
    new_text, n = REPEATED_WORDS_RE.subn(repl, text)
    return new_text, n


def _correct_unmasked(text: str) -> tuple[str, int]:
    text, dedup_count = _dedup_repeated_words(text)
    words, candidates, bigrams, trigrams = _load_lexicon()
    if not words or not candidates:
        return text, dedup_count

    matches = list(WORD_RE.finditer(text))
    replacements: dict[int, str] = {}

    def current_token(idx: int) -> str:
        return replacements.get(idx, matches[idx].group(0))

    # Pass 1: Trigram verification (highest confidence)
    if trigrams and len(matches) >= 3:
        for index in range(len(matches) - 2):
            w1_src = current_token(index)
            w2_src = current_token(index + 1)
            w3_src = current_token(index + 2)
            all_upper = w1_src.isupper() and w2_src.isupper() and w3_src.isupper()
            if w1_src.isupper() or w2_src.isupper() or w3_src.isupper():
                if not (all_upper and (_has_vn_diacritic(w1_src) or _has_vn_diacritic(w2_src) or _has_vn_diacritic(w3_src))):
                    continue
            if w1_src.istitle() and w2_src.istitle() and w3_src.istitle():
                continue
            sep1 = text[matches[index].end():matches[index + 1].start()]
            sep2 = text[matches[index + 1].end():matches[index + 2].start()]
            if not re.fullmatch(r"[ \t\r\n]+", sep1) or "\n\n" in sep1.replace("\r", ""):
                continue
            if not re.fullmatch(r"[ \t\r\n]+", sep2) or "\n\n" in sep2.replace("\r", ""):
                continue
            w1_opts = _options(w1_src, words, candidates)
            w2_opts = _options(w2_src, words, candidates)
            w3_opts = _options(w3_src, words, candidates)
            if not w1_opts or not w2_opts or not w3_opts:
                continue
            valid_tri = [
                (a, b, c) for a in w1_opts for b in w2_opts for c in w3_opts
                if (a, b, c) in trigrams
            ]
            if len(valid_tri) == 1:
                a, b, c = valid_tri[0]
                if a != w1_src.casefold():
                    replacements[index] = _case_style(matches[index].group(0), a)
                if b != w2_src.casefold():
                    replacements[index + 1] = _case_style(matches[index + 1].group(0), b)
                if c != w3_src.casefold():
                    replacements[index + 2] = _case_style(matches[index + 2].group(0), c)

    # Pass 2: Bigram verification
    if bigrams and len(matches) >= 2:
        for index in range(len(matches) - 1):
            left_source = current_token(index)
            right_source = current_token(index + 1)
            all_upper = left_source.isupper() and right_source.isupper()
            if left_source.isupper() or right_source.isupper():
                if not (all_upper and (_has_vn_diacritic(left_source) or _has_vn_diacritic(right_source))):
                    continue
            if left_source.istitle() and right_source.istitle():
                continue
            separator = text[matches[index].end():matches[index + 1].start()]
            if not re.fullmatch(r"[ \t\r\n]+", separator) or "\n\n" in separator.replace("\r", ""):
                continue
            left_options = _options(left_source, words, candidates)
            right_options = _options(right_source, words, candidates)
            if not left_options or not right_options:
                continue
            valid = [(left, right) for left in left_options for right in right_options if (left, right) in bigrams]
            if len(valid) != 1:
                continue
            left, right = valid[0]
            if left != left_source.casefold():
                replacements[index] = _case_style(matches[index].group(0), left)
            if right != right_source.casefold():
                replacements[index + 1] = _case_style(matches[index + 1].group(0), right)

    if not replacements:
        return text, dedup_count
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        pieces.append(text[cursor:match.start()])
        pieces.append(replacements.get(index, match.group(0)))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), len(replacements) + dedup_count


TABLE_SEPARATOR_ROW_RE = re.compile(r"^\s*\|?\s*(?::?-+:?\s*\|)+\s*(?::?-+:?\s*)?\|?\s*$")
CELL_SAFE_VALUE_RE = re.compile(
    r"^\s*(?:[\d.,/%+-]+|[A-ZĐ0-9._/-]{1,12}|[A-Za-z0-9._/-]+\s*=\s*\d+)?\s*$",
    re.IGNORECASE,
)


def correct_vietnamese_spelling(markdown: str) -> tuple[str, int]:
    """Correct uniquely-confirmed Vietnamese bigrams outside protected regions."""
    spans = [(match.start(), match.end()) for match in MASK_RE.finditer(markdown)]
    offset = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|"):
            if TABLE_SEPARATOR_ROW_RE.match(stripped):
                spans.append((offset, offset + len(line)))
            else:
                # Mask pipe characters
                for match in re.finditer(r"\|", line):
                    spans.append((offset + match.start(), offset + match.end()))
                # Mask numeric/code/short identifier cells
                for cell_m in re.finditer(r"[^|]+", line):
                    cell_text = cell_m.group(0)
                    if CELL_SAFE_VALUE_RE.match(cell_text):
                        spans.append((offset + cell_m.start(), offset + cell_m.end()))
        offset += len(line)
    table_start: int | None = None
    table_depth = 0
    for match in TABLE_TAG_RE.finditer(markdown):
        is_closing = match.group(0).lstrip().startswith("</")
        if not is_closing:
            if table_depth == 0:
                table_start = match.start()
            table_depth += 1
        elif table_depth:
            table_depth -= 1
            if table_depth == 0 and table_start is not None:
                spans.append((table_start, match.end()))
                table_start = None
    if table_depth and table_start is not None:
        spans.append((table_start, len(markdown)))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    output: list[str] = []
    fixed_count = 0
    cursor = 0
    for start, end in merged:
        corrected, count = _correct_unmasked(markdown[cursor:start])
        output.extend((corrected, markdown[start:end]))
        fixed_count += count
        cursor = end
    corrected, count = _correct_unmasked(markdown[cursor:])
    output.append(corrected)
    return "".join(output), fixed_count + count


def suggest_vietnamese_spelling(markdown: str) -> list[SpellingWarning]:
    """Return conservative, structured suggestions without changing Markdown."""
    corrected, _ = correct_vietnamese_spelling(markdown)
    original_words = list(WORD_RE.finditer(markdown))
    corrected_words = list(WORD_RE.finditer(corrected))
    if len(original_words) != len(corrected_words):
        return []

    warnings: list[SpellingWarning] = []
    for original, suggestion in zip(original_words, corrected_words):
        source = original.group(0)
        proposed = suggestion.group(0)
        if source == proposed:
            continue
        warnings.append({
            "word": source,
            "line": markdown.count("\n", 0, original.start()) + 1,
            "suggestion": proposed,
            "confidence": "high",
        })
    return warnings
