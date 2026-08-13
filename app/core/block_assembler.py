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
