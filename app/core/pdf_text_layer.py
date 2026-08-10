"""Native PDF text extraction used before falling back to vision OCR."""

from __future__ import annotations

from dataclasses import dataclass
import re

# pyrefly: ignore [missing-import]
import pymupdf


@dataclass(frozen=True)
class NativeTextPage:
    markdown: str
    character_count: int
    word_count: int

    @property
    def is_usable(self) -> bool:
        """A real text layer has enough content to be more reliable than OCR."""
        return self.character_count >= 40 and self.word_count >= 8


def _clean_block(text: str) -> str:
    # Preserve paragraph boundaries while undoing line wrapping in native PDF text.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return " ".join(lines)


def extract_native_text(page: pymupdf.Page) -> NativeTextPage:
    """Extract the PDF's actual text blocks in visual reading order.

    This function never invents image Markdown.  An image can only be exported by
    a separate asset extractor that has an actual PDF image object to save.
    """
    blocks = page.get_text("blocks", sort=True)
    paragraphs = [_clean_block(block[4]) for block in blocks if block[6] == 0]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    markdown = "\n\n".join(paragraphs)
    plain = re.sub(r"\s+", " ", markdown).strip()
    return NativeTextPage(
        markdown=markdown,
        character_count=len(plain),
        word_count=len(plain.split()),
    )
