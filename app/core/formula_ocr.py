"""Specialized math formula recognition using pix2tex (LaTeX-OCR)."""

from __future__ import annotations

import importlib.util
from io import BytesIO
# pyrefly: ignore [missing-import]
from PIL import Image

_model_instance = None
_initialization_error: str | None = None


def formula_ocr_status() -> tuple[bool, str]:
    """Report availability without loading the heavy pix2tex model."""
    if _model_instance is False:
        return False, f"LaTeX-OCR disabled: {_initialization_error or 'model initialization failed'}"
    if importlib.util.find_spec("pix2tex") is None:
        return False, "LaTeX-OCR disabled: optional dependency pix2tex is not installed; using Qwen fallback"
    if _model_instance is None:
        return True, "LaTeX-OCR available (model will load on first detected formula)"
    return True, "LaTeX-OCR enabled"


def get_formula_model():
    """Lazily load the LaTeX-OCR model instance."""
    global _model_instance, _initialization_error
    if _model_instance is None:
        try:
            # Import dynamically to avoid loading torch/pix2tex if formula OCR is not used
            # pyrefly: ignore [missing-import]
            from pix2tex.cli import LatexOCR
            _model_instance = LatexOCR()
        except Exception as error:
            _initialization_error = str(error)
            print(f"[Warning] Failed to initialize pix2tex: {error}")
            _model_instance = False  # Sentinel for unavailable/failed loading
    return _model_instance


def recognize_formula(image_bytes: bytes) -> str | None:
    """Predict LaTeX string from cropped formula image bytes.

    Returns the LaTeX expression or None if model is unavailable/fails.
    """
    model = get_formula_model()
    if not model or model is False:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            image = img.convert("RGB")
            latex = model(image)
            return str(latex).strip()
    except Exception as run_error:
        print(f"[Warning] Formula recognition failed: {run_error}")
        return None
