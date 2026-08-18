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
def _load_lexicon() -> tuple[frozenset[str], dict[str, tuple[str, ...]], frozenset[tuple[str, str]]]:
    """Load the replaceable JSON lexicon once per process."""
    try:
        data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return frozenset(), {}, frozenset()

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
    return words, candidates, bigrams


def _case_style(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:].lower()
    return replacement.lower() if source.islower() else replacement


def _options(token: str, words: frozenset[str], candidates: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    normalized = token.casefold()
    if normalized in words:
        return (normalized,)
    return candidates.get(_unaccented(normalized), ())


def _correct_unmasked(text: str) -> tuple[str, int]:
    words, candidates, bigrams = _load_lexicon()
    if not words or not candidates or not bigrams:
        return text, 0

    matches = list(WORD_RE.finditer(text))
    replacements: dict[int, str] = {}
    reserved: set[int] = set()
    for index in range(len(matches) - 1):
        if index in reserved or index + 1 in reserved:
            continue
        left_source = matches[index].group(0)
        right_source = matches[index + 1].group(0)
        # All-uppercase tokens commonly denote names, agencies, acronyms, or
        # document identifiers. Image-free lexical evidence is insufficient to
        # alter or flag them safely.
        if left_source.isupper() or right_source.isupper():
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
        changed = False
        if left != left_source.casefold():
            replacements[index] = _case_style(left_source, left)
            changed = True
        if right != right_source.casefold():
            replacements[index + 1] = _case_style(right_source, right)
            changed = True
        if changed:
            reserved.update((index, index + 1))

    if not replacements:
        return text, 0
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        pieces.append(text[cursor:match.start()])
        pieces.append(replacements.get(index, match.group(0)))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), len(replacements)


def correct_vietnamese_spelling(markdown: str) -> tuple[str, int]:
    """Correct uniquely-confirmed Vietnamese bigrams outside protected regions."""
    spans = [(match.start(), match.end()) for match in MASK_RE.finditer(markdown)]
    # Pipe-table cells frequently contain identifiers, names, and compact data;
    # never apply image-free spelling guesses inside them.
    offset = 0
    for line in markdown.splitlines(keepends=True):
        if line.strip().startswith("|"):
            spans.append((offset, offset + len(line)))
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
