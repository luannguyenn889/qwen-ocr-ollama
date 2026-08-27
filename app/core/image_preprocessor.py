"""Module for document image enhancement, background whitening, and contrast adaptation."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image


def is_scan_degraded(img_bgr: np.ndarray) -> bool:
    """Check if the scanned document has non-uniform lighting, yellow/brown tint, or low contrast."""
    if img_bgr is None or img_bgr.size == 0:
        return False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

    # 1. Check background brightness and contrast
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Clean digital PDFs typically have mean > 245 and standard deviation focused on black text
    if mean_val < 235 or std_val < 30:
        return True

    # 2. Check for color tint (yellowish / brownish aging in newsprint)
    if len(img_bgr.shape) == 3:
        b, g, r = cv2.split(img_bgr)
        mean_b = float(np.mean(b))
        mean_r = float(np.mean(r))
        if mean_r > 140 and mean_b < mean_r * 0.90:
            return True

    return False


def enhance_scanned_document(
    image_input: str | Path | Image.Image | np.ndarray,
    output_path: str | Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """Enhance scanned document by removing yellow/dirty background and sharpening text."""
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        with open(img_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if img is None or img.size == 0:
        return img

    if not force and not is_scan_degraded(img):
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            _, encoded = cv2.imencode(".png", img)
            with open(out_p, "wb") as f:
                f.write(encoded.tobytes())
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Step 0: Auto High-DPI Upscaling if image resolution is low (< 1500px on minimum dimension)
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 1500:
        scale = 3.0 if min_dim < 800 else 2.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = gray.shape[:2]

    # Step 1: Estimate background illumination map using morphological closing
    kernel_size = max(21, (min(h, w) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)

    # Step 2: Background division normalization (removes yellowing, shadows, dark borders)
    bg_float = background.astype(np.float32)
    bg_float[bg_float == 0] = 1.0
    normalized = (gray.astype(np.float32) / bg_float) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # Step 4: Subtle stroke sharpening to restore thin strokes (e.g. crossbar of 'Đ', diacritics)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)

    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        _, encoded = cv2.imencode(".png", result_bgr)
        with open(out_p, "wb") as f:
            f.write(encoded.tobytes())

    return result_bgr
