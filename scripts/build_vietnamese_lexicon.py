"""Build the OCR spell-correction lexicon from the GPL-2.0 Viet74K list.

Source: https://github.com/duyet/vietnamese-wordlist/blob/master/Viet74K.txt
Run from the repository root:
    python scripts/build_vietnamese_lexicon.py path/to/Viet74K.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata


TOKEN_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+$")


def unaccented(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )


def read_entries(path: Path) -> list[str]:
    entries: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = unicodedata.normalize("NFC", raw_line.strip().casefold())
        parts = entry.split()
        if 1 <= len(parts) <= 2 and all(TOKEN_RE.fullmatch(part) for part in parts):
            entries.add(" ".join(parts))
    return sorted(entries)


def build(entries: list[str]) -> dict[str, object]:
    tokens = sorted({token for entry in entries for token in entry.split()})
    groups: dict[str, list[str]] = {}
    for token in tokens:
        groups.setdefault(unaccented(token), []).append(token)
    candidates = {
        plain: sorted(set(values))
        for plain, values in sorted(groups.items())
        if any(value != plain for value in values)
    }
    return {
        "source": {
            "name": "Viet74K",
            "url": "https://github.com/duyet/vietnamese-wordlist",
            "license": "GPL-2.0",
        },
        "words": tokens,
        "accent_candidates": candidates,
        "bigrams": [entry for entry in entries if len(entry.split()) == 2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vietnamese_lexicon.json from Viet74K.txt")
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("app/core/vietnamese_lexicon.json"),
    )
    args = parser.parse_args()
    entries = read_entries(args.source)
    payload = build(entries)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(payload['words'])} tokens, "
        f"{len(payload['accent_candidates'])} accent groups, "
        f"and {len(payload['bigrams'])} bigrams to {args.output}"
    )


if __name__ == "__main__":
    main()
