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
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Load the replaceable JSON lexicon once per process."""
    try:
        data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return frozenset(), {}, frozenset(), frozenset(), {}, {}

    words = frozenset(str(word).casefold() for word in data.get("words", []))
    candidates: dict[str, tuple[str, ...]] = {}
    for plain, accented in data.get("accent_candidates", {}).items():
        values = tuple(dict.fromkeys(str(word).casefold() for word in accented))
        if values:
            candidates[_unaccented(str(plain))] = values
    bigrams_list = [
        (parts[0].casefold(), parts[1].casefold())
        for item in data.get("bigrams", [])
        if len(parts := str(item).split()) == 2
    ]
    bigrams = frozenset(bigrams_list)
    trigrams = frozenset(
        (parts[0].casefold(), parts[1].casefold(), parts[2].casefold())
        for item in data.get("trigrams", [])
        if len(parts := str(item).split()) == 3
    )
    bg_left_temp: dict[str, list[str]] = {}
    bg_right_temp: dict[str, list[str]] = {}
    for l_p, r_p in bigrams_list:
        bg_left_temp.setdefault(l_p, []).append(r_p)
        bg_right_temp.setdefault(r_p, []).append(l_p)
    bg_left = {k: tuple(v) for k, v in bg_left_temp.items()}
    bg_right = {k: tuple(v) for k, v in bg_right_temp.items()}

    return words, candidates, bigrams, trigrams, bg_left, bg_right


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings (optimized for <= 1)."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 1:
        return 2
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if len1 == len2:
        return sum(1 for a, b in zip(s1, s2) if a != b)
    shorter, longer = (s1, s2) if len1 < len2 else (s2, s1)
    i = j = diff = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            diff += 1
            if diff > 1:
                return 2
            j += 1
        else:
            i += 1
            j += 1
    return 1


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
    nfd = unicodedata.normalize("NFD", value)
    return any(unicodedata.category(c) == "Mn" for c in nfd) or any(c.casefold() in ("đ",) for c in value)


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
    words, candidates, bigrams, trigrams, bg_left, bg_right = _load_lexicon()
    if not words or not candidates:
        return text, dedup_count

    matches = list(WORD_RE.finditer(text))
    replacements: dict[int, str] = {}

    # Pass 0: Isolated non-word diacritic repair
    for index, match in enumerate(matches):
        raw_token = match.group(0)
        token_cf = raw_token.casefold()
        if len(token_cf) < 3:
            continue
        if token_cf in words:
            continue
        if raw_token.isupper() and not _has_vn_diacritic(raw_token):
            continue

        unacc = _unaccented(token_cf)
        pool = candidates.get(unacc, ())
        if not pool:
            continue

        if len(pool) == 1:
            replacements[index] = _case_style(raw_token, pool[0])
        else:
            def vowel_core(s: str) -> str:
                return "".join(c for c in unicodedata.normalize("NFD", s) if c in "ăâêôơưđĂÂÊÔƠƯĐ")

            target_vowel = vowel_core(token_cf)
            matching_vowels = [c for c in pool if vowel_core(c) == target_vowel]
            if len(matching_vowels) == 1:
                replacements[index] = _case_style(raw_token, matching_vowels[0])

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
            w1_cf, w2_cf, w3_cf = w1_src.casefold(), w2_src.casefold(), w3_src.casefold()
            if (w1_cf, w2_cf, w3_cf) in trigrams:
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
                if a != w1_cf and (w1_cf not in words or (w1_cf, b) not in bigrams):
                    replacements[index] = _case_style(matches[index].group(0), a)
                if b != w2_cf and (w2_cf not in words or (a, w2_cf) not in bigrams):
                    replacements[index + 1] = _case_style(matches[index + 1].group(0), b)
                if c != w3_cf and (w3_cf not in words or (b, w3_cf) not in bigrams):
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
            l_cf, r_cf = left_source.casefold(), right_source.casefold()
            if (l_cf, r_cf) in bigrams:
                continue

            left_options = _options(left_source, words, candidates)
            right_options = _options(right_source, words, candidates)
            if not left_options or not right_options:
                continue
            valid = [(left, right) for left in left_options for right in right_options if (left, right) in bigrams]
            if len(valid) != 1:
                continue
            left, right = valid[0]
            # Only change a word if it is not in words OR if current word is a clear non-word
            if left != l_cf and (l_cf not in words or r_cf not in words):
                replacements[index] = _case_style(matches[index].group(0), left)
            if right != r_cf and (r_cf not in words or l_cf not in words):
                replacements[index + 1] = _case_style(matches[index + 1].group(0), right)

    # Pass 3: Fuzzy Bigram Verification (Levenshtein Edit Distance <= 1)
    stopwords = frozenset({"và", "là", "của", "trong", "được", "có", "với", "đã", "đang", "sẽ", "cho", "về", "tại", "từ", "bởi", "do", "nhưng", "mà", "hoặc", "hay", "để", "vì", "nếu", "thì", "như", "các", "những", "mỗi", "mọi", "này", "đó", "kia"})
    bound_tokens = set()
    for idx in range(len(matches) - 1):
        if (current_token(idx).casefold(), current_token(idx + 1).casefold()) in bigrams:
            bound_tokens.add(idx)
            bound_tokens.add(idx + 1)

    if bigrams and len(matches) >= 2:
        for index in range(len(matches) - 1):
            l_src = current_token(index)
            r_src = current_token(index + 1)
            if (l_src.isupper() and not _has_vn_diacritic(l_src) and len(l_src) <= 4) or (r_src.isupper() and not _has_vn_diacritic(r_src) and len(r_src) <= 4):
                continue
            l_cf = l_src.casefold()
            r_cf = r_src.casefold()
            if (l_cf, r_cf) in bigrams:
                continue
            if l_cf in stopwords or r_cf in stopwords:
                continue

            cand_pairs: list[tuple[int, str, str]] = []
            if r_cf in bg_right:
                for cand_l in bg_right[r_cf]:
                    if cand_l not in stopwords:
                        dist = _levenshtein_distance(l_cf, cand_l)
                        if dist <= 1:
                            cand_pairs.append((dist, cand_l, r_cf))

            if l_cf in bg_left:
                for cand_r in bg_left[l_cf]:
                    if cand_r not in stopwords:
                        dist = _levenshtein_distance(r_cf, cand_r)
                        if dist <= 1:
                            cand_pairs.append((dist, l_cf, cand_r))

            if not cand_pairs:
                unacc_l = _unaccented(l_cf)
                pool_l = candidates.get(unacc_l, (l_cf,))
                for cand_l in pool_l:
                    if cand_l in bg_left:
                        for cand_r in bg_left[cand_l]:
                            if cand_l not in stopwords and cand_r not in stopwords:
                                d_l = _levenshtein_distance(l_cf, cand_l)
                                d_r = _levenshtein_distance(r_cf, cand_r)
                                if d_l <= 1 and d_r <= 1 and (d_l + d_r) <= 2:
                                    cand_pairs.append((d_l + d_r, cand_l, cand_r))

            if cand_pairs:
                cand_pairs.sort(key=lambda x: x[0])
                min_dist = cand_pairs[0][0]
                best_candidates = [p for p in cand_pairs if p[0] == min_dist]
                if len(best_candidates) == 1 or (len(best_candidates) > 1 and best_candidates[0][1] == best_candidates[1][1]):
                    _, best_l, best_r = best_candidates[0]
                    if index not in bound_tokens and best_l != l_cf:
                        replacements[index] = _case_style(matches[index].group(0), best_l)
                    if (index + 1) not in bound_tokens and best_r != r_cf:
                        replacements[index + 1] = _case_style(matches[index + 1].group(0), best_r)

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
