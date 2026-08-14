"""Backward-compatible import; rendering now lives in ``pdf_renderer``."""

from app.core.pdf_renderer import render_pdf

__all__ = ["render_pdf"]
