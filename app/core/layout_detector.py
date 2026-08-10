"""PaddleOCR layout helpers.  This module never reads or returns OCR text."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

BoundingBox = tuple[float, float, float, float]


def _result_data(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "res"):
        data = result.res
    elif hasattr(result, "to_dict"):
        data = result.to_dict()
    else:
        data = dict(result)
    return data.get("res", data)


def _image_array(image_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))


def detect_columns(regions: list[BoundingBox], image_width: int) -> list[tuple[int, int]]:
    """Return left-to-right column spans from detected text boxes.

    A large gap between adjacent horizontal box centres separates columns.
    Small/isolated boxes are ignored so titles spanning the full page do not
    create a false column.  One span means the caller must keep the full page.
    """
    usable = [box for box in regions if box[2] > box[0] and box[3] > box[1]
              and (box[2] - box[0]) < image_width * 0.92]
    if len(usable) < 4:
        return [(0, image_width)]
    centres = sorted((left + right) / 2 for left, _, right, _ in usable)
    gaps = [(centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)]
    min_gap = max(image_width * 0.12, 40)
    split_gaps = [(gap, idx) for gap, idx in gaps if gap >= min_gap]
    if not split_gaps:
        return [(0, image_width)]

    # Keep at most two strongest splits (three columns); each side needs boxes.
    split_indices = sorted(idx for _, idx in sorted(split_gaps, reverse=True)[:2])
    boundaries = [0] + [int((centres[i] + centres[i + 1]) / 2) for i in split_indices] + [image_width]
    spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return spans if all(right - left >= image_width * 0.18 for left, right in spans) else [(0, image_width)]


@dataclass
class LayoutDetector:
    """Load detection and PP-DocLayout once; never instantiate recognition."""

    _text_detector: Any = None
    _layout_detector: Any = None

    def __post_init__(self) -> None:
        from paddleocr import LayoutDetection, TextDetection

        self._text_detector = TextDetection()
        # PaddleOCR 3.x maps this default to the PP-DocLayout model family.
        self._layout_detector = LayoutDetection()

    def detect_text_regions(self, image_bytes: bytes) -> list[BoundingBox]:
        results = self._text_detector.predict(_image_array(image_bytes))
        boxes: list[BoundingBox] = []
        for result in results:
            data = _result_data(result)
            for polygon in data.get("dt_polys", []):
                points = np.asarray(polygon)
                boxes.append((float(points[:, 0].min()), float(points[:, 1].min()),
                              float(points[:, 0].max()), float(points[:, 1].max())))
        return boxes

    def detect_has_table(self, image_bytes: bytes) -> bool:
        return bool(self.detect_layout_regions(image_bytes)[1])

    def detect_layout_regions(self, image_bytes: bytes) -> tuple[list[BoundingBox], list[BoundingBox]]:
        """Return layout geometry for all regions and the table subset only."""
        all_regions: list[BoundingBox] = []
        tables: list[BoundingBox] = []
        for result in self._layout_detector.predict(_image_array(image_bytes)):
            data = _result_data(result)
            boxes = data.get("boxes", data.get("dt_polys", []))
            labels = data.get("labels", data.get("label_names", []))
            for index, box in enumerate(boxes):
                # PP-DocLayout returns a list of dictionaries such as
                # {"label": "text", "coordinate": [x0, y0, x1, y1]}.
                # Other Paddle models return polygons/arrays directly.
                item_label = labels[index] if index < len(labels) else ""
                if isinstance(box, dict):
                    item_label = box.get("label", item_label)
                    points = np.asarray(box.get("coordinate", box.get("bbox", [])))
                else:
                    points = np.asarray(box)
                if points.size < 4:
                    continue
                if points.ndim == 1 and len(points) == 4:
                    left, top, right, bottom = map(float, points)
                else:
                    left, top = float(points[:, 0].min()), float(points[:, 1].min())
                    right, bottom = float(points[:, 0].max()), float(points[:, 1].max())
                region = (left, top, right, bottom)
                all_regions.append(region)
                if str(item_label).casefold() == "table":
                    tables.append(region)
        return all_regions, tables

    def analyse(self, image_path: str | Path) -> tuple[list[tuple[int, int]], bool]:
        image_bytes = Path(image_path).read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            width = image.width
        return detect_columns(self.detect_text_regions(image_bytes), width), self.detect_has_table(image_bytes)

    def analyse_with_regions(self, image_path: str | Path) -> tuple[list[tuple[int, int]], bool, list[BoundingBox], list[BoundingBox]]:
        image_bytes = Path(image_path).read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            width = image.width
        text_regions = self.detect_text_regions(image_bytes)
        layout_regions, table_regions = self.detect_layout_regions(image_bytes)
        return detect_columns(text_regions, width), bool(table_regions), text_regions, table_regions


def crop_columns(image_path: str | Path, columns: list[tuple[int, int]], output_dir: str | Path) -> list[Path]:
    """Save crops in left-to-right order.  A full-width span is not cropped."""
    source = Path(image_path)
    with Image.open(source) as image:
        if len(columns) <= 1:
            return [source]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        crops: list[Path] = []
        for number, (left, right) in enumerate(columns, 1):
            crop_path = output / f"{source.stem}_column_{number}.png"
            image.crop((left, 0, right, image.height)).save(crop_path)
            crops.append(crop_path)
    return crops


# Convenience API for callers that only need one operation.  The process-wide
# instance keeps both Paddle models loaded exactly once.
_default_detector: LayoutDetector | None = None


def _default() -> LayoutDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = LayoutDetector()
    return _default_detector


def detect_text_regions(image_bytes: bytes) -> list[BoundingBox]:
    """Detect text geometry only; no recognised strings are exposed."""
    return _default().detect_text_regions(image_bytes)


def detect_has_table(image_bytes: bytes) -> bool:
    """Return whether PP-DocLayout labels at least one region as a table."""
    return _default().detect_has_table(image_bytes)


def save_layout_overlay(
    image_path: str | Path, output_path: str | Path, text_regions: list[BoundingBox],
    columns: list[tuple[int, int]], table_regions: list[BoundingBox],
) -> Path:
    """Create a diagnostic image; it never alters the OCR input/output image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in text_regions:
        draw.rectangle((left, top, right, bottom), outline="#1688ff", width=2)
    for left, right in columns[1:]:
        draw.line((left, 0, left, image.height), fill="#ff9800", width=5)
    for left, top, right, bottom in table_regions:
        draw.rectangle((left, top, right, bottom), outline="#e53935", width=4)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
