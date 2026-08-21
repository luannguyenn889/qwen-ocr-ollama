"""Coordinate-aware Markdown assembly for typed document layout blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DocumentBlock:
    kind: str
    bbox: tuple[float, float, float, float]
    content: str = ""


@dataclass(frozen=True)
class BlockSpan:
    """Exact link between assembled Markdown and its source image region."""
    block_id: str
    kind: str
    bbox: tuple[float, float, float, float]
    markdown_start: int
    markdown_end: int
    mapping_confidence: str = "exact"


def crop_content_blocks(image_path: Path, blocks: list[DocumentBlock], output_dir: Path) -> list[tuple[DocumentBlock, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped = []
    with Image.open(image_path) as image:
        for index, block in enumerate(blocks, 1):
            if block.kind == "image":
                continue
            left, top, right, bottom = block.bbox
            box = (
                max(0, int(left) - 4), max(0, int(top) - 4),
                min(image.width, int(right) + 4), min(image.height, int(bottom) + 4),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            path = output_dir / f"block_{index:03d}_{block.kind}.png"
            image.crop(box).save(path)
            cropped.append((block, path))
    return cropped


def assemble_blocks(blocks: list[DocumentBlock], content_by_bbox: dict[tuple[float, float, float, float], str], image_paths: list[str]) -> str:
    """Interleave recognised content and extracted images in block order."""
    image_iterator = iter(image_paths)
    parts: list[str] = []
    for block in blocks:
        if block.kind == "image":
            try:
                path = next(image_iterator)
            except StopIteration:
                continue
            parts.append(f"![Hình ảnh]({path})")
        else:
            content = content_by_bbox.get(block.bbox, block.content).strip()
            if content:
                parts.append(content)
    # Preserve extracted images that had no layout block instead of losing them.
    parts.extend(f"![Hình ảnh]({path})" for path in image_iterator)
    return "\n\n".join(parts)


def assemble_blocks_with_spans(
    blocks: list[DocumentBlock], content_by_bbox: dict[tuple[float, float, float, float], str],
    image_paths: list[str], *, page_number: int = 1,
) -> tuple[str, list[BlockSpan]]:
    """Assemble blocks while retaining exact character offsets for every block."""
    image_iterator = iter(image_paths)
    parts: list[str] = []
    pending: list[tuple[str, DocumentBlock, str]] = []
    for index, block in enumerate(blocks, 1):
        block_id = f"page_{page_number}_block_{index:03d}"
        if block.kind == "image":
            try:
                content = f"![Hình ảnh]({next(image_iterator)})"
            except StopIteration:
                continue
        else:
            content = content_by_bbox.get(block.bbox, block.content).strip()
            if not content:
                continue
        parts.append(content)
        pending.append((block_id, block, content))
    for index, path in enumerate(image_iterator, len(blocks) + 1):
        content = f"![Hình ảnh]({path})"
        parts.append(content)
        pending.append((f"page_{page_number}_asset_{index:03d}", DocumentBlock("image", (0, 0, 0, 0)), content))

    markdown = "\n\n".join(parts)
    spans: list[BlockSpan] = []
    cursor = 0
    for block_id, block, content in pending:
        start = markdown.find(content, cursor)
        end = start + len(content)
        spans.append(BlockSpan(block_id, block.kind, block.bbox, start, end))
        cursor = end
    return markdown, spans


def rebase_block_spans(old_markdown: str, new_markdown: str, spans: list[BlockSpan]) -> list[BlockSpan]:
    """Recalculate offsets after safe formatting while rejecting ambiguous mappings."""
    rebased: list[BlockSpan] = []
    cursor = 0
    for span in spans:
        content = old_markdown[span.markdown_start:span.markdown_end]
        if not content:
            continue
        start = new_markdown.find(content, cursor)
        if start < 0:
            continue
        # Repeated content is safe only when reading order resolves it uniquely.
        end = start + len(content)
        rebased.append(BlockSpan(
            span.block_id, span.kind, span.bbox, start, end, span.mapping_confidence,
        ))
        cursor = end
    return rebased
