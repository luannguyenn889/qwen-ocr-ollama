"""PaddleOCR layout helpers.  This module never reads or returns OCR text."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
from typing import Any

# PaddleX otherwise probes public model hosters even when cached/local models are
# intended. Set this before importing paddleocr in LayoutDetector.__post_init__.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "3")

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw

BoundingBox = tuple[float, float, float, float]


def detect_columns(regions: list[BoundingBox], image_width: int) -> list[tuple[int, int]]:
    """Compatibility helper: split clearly separated horizontal region groups."""
    if not regions:
        return [(0, image_width)]
    intervals = sorted((box[0], box[2]) for box in regions)
    merged: list[list[float]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1] + image_width * 0.05:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    if len(merged) <= 1:
        return [(0, image_width)]
    boundaries = [0]
    for current, following in zip(merged, merged[1:]):
        boundaries.append(min(image_width, int(round((current[1] + following[0]) / 2)) + 2))
    boundaries.append(image_width)
    return list(zip(boundaries, boundaries[1:]))


@dataclass(frozen=True)
class PageLayoutAnalysis:
    segments: list[BoundingBox]
    blocks: list[tuple[str, BoundingBox]]
    tables: list[BoundingBox]
    images: list[BoundingBox]
    formulas: list[BoundingBox]
    regions: list[BoundingBox]


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


def _resize_image_bytes(image_bytes: bytes, max_width: int = 800) -> tuple[bytes, float, float]:
    """Resize image to max_width preserving aspect ratio. Return (resized_bytes, scale_x, scale_y)."""
    with Image.open(BytesIO(image_bytes)) as img:
        orig_w, orig_h = img.width, img.height
        if orig_w <= max_width:
            return image_bytes, 1.0, 1.0

        new_w = max_width
        new_h = int(orig_h * (max_width / orig_w))

        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        resized_img = img.resize((new_w, new_h), resample_filter)

        bio = BytesIO()
        resized_img.save(bio, format="PNG")

        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        return bio.getvalue(), scale_x, scale_y


@lru_cache(maxsize=16)
def _predict_layout_cached(image_bytes: bytes, predict_fn: Any) -> tuple[list[dict[str, Any]], float, float]:
    resized_bytes, scale_x, scale_y = _resize_image_bytes(image_bytes, 800)
    results = predict_fn(_image_array(resized_bytes))
    cached_data = []
    for result in results:
        data = _result_data(result)
        boxes = data.get("boxes", data.get("dt_polys", []))
        labels = data.get("labels", data.get("label_names", []))
        converted_boxes = []
        for box in boxes:
            if isinstance(box, dict):
                coord = box.get("coordinate", box.get("bbox", []))
                converted_boxes.append({
                    "label": box.get("label", ""),
                    "coordinate": [float(v) for v in np.asarray(coord).flatten()]
                })
            else:
                converted_boxes.append(np.asarray(box).tolist())
        cached_data.append({
            "boxes": converted_boxes,
            "labels": [str(l) for l in labels]
        })
    return cached_data, scale_x, scale_y


def recursive_xy_cut(boxes: list[BoundingBox], min_x_gap: float = 15.0, min_y_gap: float = 15.0) -> list[BoundingBox]:
    if not boxes:
        return []
    if len(boxes) == 1:
        return boxes

    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)

    # Try Y cut first
    y_sorted = sorted(boxes, key=lambda b: b[1])
    max_y = y_sorted[0][3]
    best_y_split = None

    for i in range(1, len(y_sorted)):
        box = y_sorted[i]
        if box[1] > max_y + min_y_gap:
            best_y_split = max_y + (box[1] - max_y) / 2
            break
        max_y = max(max_y, box[3])

    if best_y_split is not None:
        top_boxes = [b for b in boxes if b[3] <= best_y_split]
        bottom_boxes = [b for b in boxes if b[1] > best_y_split]
        return recursive_xy_cut(top_boxes, min_x_gap, min_y_gap) + recursive_xy_cut(bottom_boxes, min_x_gap, min_y_gap)

    # Try X cut
    x_sorted = sorted(boxes, key=lambda b: b[0])
    max_x = x_sorted[0][2]
    best_x_split = None

    for i in range(1, len(x_sorted)):
        box = x_sorted[i]
        if box[0] > max_x + min_x_gap:
            best_x_split = max_x + (box[0] - max_x) / 2
            break
        max_x = max(max_x, box[2])

    if best_x_split is not None:
        left_boxes = [b for b in boxes if b[2] <= best_x_split]
        right_boxes = [b for b in boxes if b[0] > best_x_split]
        return recursive_xy_cut(left_boxes, min_x_gap, min_y_gap) + recursive_xy_cut(right_boxes, min_x_gap, min_y_gap)

    # If no cuts, return the bounding box of the remaining cluster
    return [(left, top, right, bottom)]


def merge_column_blocks(boxes: list[BoundingBox], image_width: int, image_height: int) -> list[BoundingBox]:
    if not boxes:
        return [(0.0, 0.0, float(image_width), float(image_height))]
    merged = [boxes[0]]
    for current in boxes[1:]:
        prev = merged[-1]
        # Merge if they are vertically adjacent and have similar X bounds (e.g., belong to same column)
        # Tolerance for column width/alignment
        if (abs(current[0] - prev[0]) < image_width * 0.1 and
            abs(current[2] - prev[2]) < image_width * 0.1 and
            current[1] >= prev[3] - 20):
            merged[-1] = (
                min(prev[0], current[0]),
                min(prev[1], current[1]),
                max(prev[2], current[2]),
                max(prev[3], current[3])
            )
        else:
            merged.append(current)

    # Expand slightly to avoid cutting text
    padded = []
    pad = 5.0
    for (l, t, r, b) in merged:
        padded.append((
            max(0.0, l - pad),
            max(0.0, t - pad),
            min(float(image_width), r + pad),
            min(float(image_height), b + pad)
        ))
    return padded


def detect_reading_segments(regions: list[BoundingBox], image_width: int, image_height: int) -> list[BoundingBox]:
    """Return segments in top-to-bottom, left-to-right reading order using XY-Cut."""
    usable = [box for box in regions if box[2] > box[0] and box[3] > box[1]]
    if not usable:
        return [(0.0, 0.0, float(image_width), float(image_height))]

    # Apply recursive XY-cut
    cut_boxes = recursive_xy_cut(usable, min_x_gap=max(image_width * 0.05, 20.0), min_y_gap=15.0)

    # Merge vertically adjacent blocks in the same column
    final_segments = merge_column_blocks(cut_boxes, image_width, image_height)
    return final_segments


@dataclass
class LayoutDetector:
    """Load detection and PP-DocLayout once; never instantiate recognition."""

    _text_detector: Any = None
    _layout_detector: Any = None

    def __post_init__(self) -> None:
        # pyrefly: ignore [missing-import]
        from paddleocr import LayoutDetection, TextDetection

        text_model_dir = os.environ.get("QWEN_OCR_TEXT_DETECTION_MODEL_DIR")
        layout_model_dir = os.environ.get("QWEN_OCR_LAYOUT_MODEL_DIR")
        text_options = {"model_dir": text_model_dir} if text_model_dir else {}
        layout_options = {"model_dir": layout_model_dir} if layout_model_dir else {}

        self._text_detector = TextDetection(**text_options)
        # PaddleOCR 3.x maps this default to the PP-DocLayout model family.
        self._layout_detector = LayoutDetection(**layout_options)

    @property
    def availability(self) -> tuple[bool, str]:
        if self._text_detector is None or self._layout_detector is None:
            return False, "layout models are not initialized"
        return True, "layout enabled (offline model-source checks disabled)"

    def detect_text_regions(self, image_bytes: bytes) -> list[BoundingBox]:
        resized_bytes, scale_x, scale_y = _resize_image_bytes(image_bytes, 800)
        results = self._text_detector.predict(_image_array(resized_bytes))
        boxes: list[BoundingBox] = []
        for result in results:
            data = _result_data(result)
            for polygon in data.get("dt_polys", []):
                points = np.asarray(polygon)
                left = float(points[:, 0].min()) * scale_x
                top = float(points[:, 1].min()) * scale_y
                right = float(points[:, 0].max()) * scale_x
                bottom = float(points[:, 1].max()) * scale_y
                boxes.append((left, top, right, bottom))
        return boxes

    def detect_has_table(self, image_bytes: bytes) -> bool:
        return bool(self.detect_layout_regions(image_bytes)[1])

    def detect_layout_regions(self, image_bytes: bytes) -> tuple[list[BoundingBox], list[BoundingBox]]:
        """Return layout geometry for all regions and the table subset only."""
        cached_data, scale_x, scale_y = _predict_layout_cached(image_bytes, self._layout_detector.predict)
        all_regions: list[BoundingBox] = []
        tables: list[BoundingBox] = []
        for result in cached_data:
            boxes = result["boxes"]
            labels = result["labels"]
            for index, box in enumerate(boxes):
                # PP-DocLayout returns a list of dictionaries such as
                # {"label": "text", "coordinate": [x0, y0, x1, y1]}.
                # Other Paddle models return polygons/arrays directly.
                item_label = labels[index] if index < len(labels) else ""
                if isinstance(box, dict):
                    item_label = box.get("label", item_label)
                    points = np.asarray(box.get("coordinate"))
                else:
                    points = np.asarray(box)
                if points.size < 4:
                    continue
                if points.ndim == 1 and len(points) == 4:
                    left, top, right, bottom = map(float, points)
                else:
                    left, top = float(points[:, 0].min()), float(points[:, 1].min())
                    right, bottom = float(points[:, 0].max()), float(points[:, 1].max())

                # Scale coordinates
                left *= scale_x
                top *= scale_y
                right *= scale_x
                bottom *= scale_y

                region = (left, top, right, bottom)
                all_regions.append(region)
                if str(item_label).casefold() == "table":
                    tables.append(region)
        return all_regions, tables

    def detect_ordered_blocks(self, image_bytes: bytes) -> list[tuple[str, BoundingBox]]:
        """Return typed layout blocks in column-aware reading order."""
        cached_data, scale_x, scale_y = _predict_layout_cached(image_bytes, self._layout_detector.predict)
        blocks: list[tuple[str, BoundingBox]] = []
        for result in cached_data:
            for index, box in enumerate(result["boxes"]):
                label = result["labels"][index] if index < len(result["labels"]) else "text"
                if isinstance(box, dict):
                    label = box.get("label", label)
                    points = np.asarray(box.get("coordinate"))
                else:
                    points = np.asarray(box)
                if points.size < 4:
                    continue
                if points.ndim == 1 and len(points) == 4:
                    left, top, right, bottom = map(float, points)
                else:
                    left, top = float(points[:, 0].min()), float(points[:, 1].min())
                    right, bottom = float(points[:, 0].max()), float(points[:, 1].max())
                bbox = (left * scale_x, top * scale_y, right * scale_x, bottom * scale_y)
                kind = str(label).casefold()
                if kind in {"figure", "image", "chart", "seal"}:
                    kind = "image"
                elif kind == "table":
                    kind = "table"
                elif kind in {"title", "heading", "header"}:
                    kind = "heading"
                elif kind in {"formula", "equation", "inline_formula", "display_formula"}:
                    kind = "formula"
                else:
                    kind = "text"
                blocks.append((kind, bbox))

        if not blocks:
            return []
        page_width = max(block[1][2] for block in blocks)
        full_width = sorted(
            (block for block in blocks if block[1][2] - block[1][0] >= page_width * 0.65),
            key=lambda block: block[1][1],
        )
        # Full-width blocks split the page into vertical bands. Inside each band,
        # read the complete left column before the right column.
        def reading_key(item):
            _, (left, top, right, bottom) = item
            width = right - left
            band = sum(full[1][3] <= top for full in full_width)
            if width >= page_width * 0.65:
                return (band, -1, top, left)
            # Quantize by detected horizontal position instead of forcing every
            # page into two columns. This preserves 3+ uneven columns.
            column_width = max(page_width * 0.18, 1.0)
            column = int(left / column_width)
            return (band, column, top, left)
        return sorted(blocks, key=reading_key)

    def analyse_page(self, image_path: str | Path) -> PageLayoutAnalysis:
        """Run PP-DocLayout once and derive every page-level layout view."""
        image_bytes = Path(image_path).read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.width, image.height
        blocks = self.detect_ordered_blocks(image_bytes)
        regions = [bbox for _, bbox in blocks]
        tables = [bbox for kind, bbox in blocks if kind == "table"]
        images = [bbox for kind, bbox in blocks if kind == "image"]
        formulas = [bbox for kind, bbox in blocks if kind == "formula"]
        return PageLayoutAnalysis(
            segments=detect_reading_segments(regions, width, height),
            blocks=blocks,
            tables=tables,
            images=images,
            formulas=formulas,
            regions=regions,
        )

    def detect_layout_tables_and_images(self, image_bytes: bytes) -> tuple[list[BoundingBox], list[BoundingBox]]:
        """Return tables and images bounding boxes respectively."""
        cached_data, scale_x, scale_y = _predict_layout_cached(image_bytes, self._layout_detector.predict)
        tables: list[BoundingBox] = []
        images: list[BoundingBox] = []
        for result in cached_data:
            boxes = result["boxes"]
            labels = result["labels"]
            for index, box in enumerate(boxes):
                item_label = labels[index] if index < len(labels) else ""
                if isinstance(box, dict):
                    item_label = box.get("label", item_label)
                    points = np.asarray(box.get("coordinate"))
                else:
                    points = np.asarray(box)
                if points.size < 4:
                    continue
                if points.ndim == 1 and len(points) == 4:
                    left, top, right, bottom = map(float, points)
                else:
                    left, top = float(points[:, 0].min()), float(points[:, 1].min())
                    right, bottom = float(points[:, 0].max()), float(points[:, 1].max())

                # Scale coordinates
                left *= scale_x
                top *= scale_y
                right *= scale_x
                bottom *= scale_y

                region = (left, top, right, bottom)
                if str(item_label).casefold() == "table":
                    tables.append(region)
                elif str(item_label).casefold() in ("image", "figure"):
                    images.append(region)
        return tables, images

    def detect_layout_tables_images_and_formulas(self, image_bytes: bytes) -> tuple[list[BoundingBox], list[BoundingBox], list[BoundingBox]]:
        """Return tables, images, and formulas bounding boxes respectively."""
        cached_data, scale_x, scale_y = _predict_layout_cached(image_bytes, self._layout_detector.predict)
        tables: list[BoundingBox] = []
        images: list[BoundingBox] = []
        formulas: list[BoundingBox] = []
        for result in cached_data:
            boxes = result["boxes"]
            labels = result["labels"]
            for index, box in enumerate(boxes):
                item_label = labels[index] if index < len(labels) else ""
                if isinstance(box, dict):
                    item_label = box.get("label", item_label)
                    points = np.asarray(box.get("coordinate"))
                else:
                    points = np.asarray(box)
                if points.size < 4:
                    continue
                if points.ndim == 1 and len(points) == 4:
                    left, top, right, bottom = map(float, points)
                else:
                    left, top = float(points[:, 0].min()), float(points[:, 1].min())
                    right, bottom = float(points[:, 0].max()), float(points[:, 1].max())

                # Scale coordinates
                left *= scale_x
                top *= scale_y
                right *= scale_x
                bottom *= scale_y

                region = (left, top, right, bottom)
                label_lower = str(item_label).casefold()
                if label_lower == "table":
                    tables.append(region)
                elif label_lower in ("image", "figure"):
                    images.append(region)
                elif label_lower in ("formula", "equation", "inline_formula", "display_formula"):
                    formulas.append(region)
        return tables, images, formulas

    def analyse(self, image_path: str | Path) -> tuple[list[BoundingBox], bool]:
        image_bytes = Path(image_path).read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.width, image.height
        layout_regions, table_regions = self.detect_layout_regions(image_bytes)
        return detect_reading_segments(layout_regions, width, height), self.detect_has_table(image_bytes)

    def analyse_with_regions(self, image_path: str | Path) -> tuple[list[BoundingBox], bool, list[BoundingBox], list[BoundingBox]]:
        image_bytes = Path(image_path).read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.width, image.height
        text_regions = self.detect_text_regions(image_bytes)
        layout_regions, table_regions = self.detect_layout_regions(image_bytes)

        all_regions = text_regions + layout_regions
        segments = detect_reading_segments(all_regions, width, height)
        return segments, bool(table_regions), text_regions, table_regions


def crop_segments(image_path: str | Path, segments: list[BoundingBox], output_dir: str | Path) -> list[Path]:
    """Save crops in reading order by stitching them vertically into a single image to prevent LLM hallucination on small chunks."""
    source = Path(image_path)
    with Image.open(source) as image:
        if len(segments) <= 1:
            return [source]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        crops = []
        for left, top, right, bottom in segments:
            crop_box = (int(left), int(top), int(right), int(bottom))
            crops.append(image.crop(crop_box))

        # Stitch vertically
        total_height = sum(c.height for c in crops) + 20 * (len(crops) - 1)
        max_width = max(c.width for c in crops)

        stitched = Image.new('RGB', (max_width, total_height), 'white')
        y_offset = 0
        for c in crops:
            stitched.paste(c, (0, y_offset))
            y_offset += c.height + 20

        stitched_path = output / f"{source.stem}_stitched_segments.png"
        stitched.save(stitched_path)
        return [stitched_path]


# Convenience API for callers that only need one operation.  The process-wide
# instance keeps both Paddle models loaded exactly once.
_default_detector: LayoutDetector | None = None


def _default() -> LayoutDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = LayoutDetector()
    return _default_detector


def create_layout_detector() -> tuple[LayoutDetector | None, str]:
    """Initialize layout detection once and return an explicit availability status."""
    if os.environ.get("QWEN_OCR_DISABLE_LAYOUT", "").casefold() in {"1", "true", "yes", "on"}:
        return None, "layout disabled by QWEN_OCR_DISABLE_LAYOUT"
    try:
        detector = LayoutDetector()
        available, message = detector.availability
        return (detector if available else None), message
    except Exception as error:
        return None, f"layout disabled: {error}"


def detect_text_regions(image_bytes: bytes) -> list[BoundingBox]:
    """Detect text geometry only; no recognised strings are exposed."""
    return _default().detect_text_regions(image_bytes)


def detect_has_table(image_bytes: bytes) -> bool:
    """Return whether PP-DocLayout labels at least one region as a table."""
    return _default().detect_has_table(image_bytes)


def save_layout_overlay(
    image_path: str | Path, output_path: str | Path, text_regions: list[BoundingBox],
    segments: list[BoundingBox], table_regions: list[BoundingBox],
) -> Path:
    """Create a diagnostic image; it never alters the OCR input/output image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    # Import ImageFont for larger font size
    try:
        # pyrefly: ignore [missing-import]
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = None

    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in text_regions:
        draw.rectangle((left, top, right, bottom), outline="#1688ff", width=2)
    for number, (left, top, right, bottom) in enumerate(segments, 1):
        draw.rectangle((left, top, right, bottom), outline="#ff9800", width=6)
        draw.text((left + 10, top + 10), str(number), fill="#ff9800", font=font)
    for left, top, right, bottom in table_regions:
        draw.rectangle((left, top, right, bottom), outline="#e53935", width=4)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
