"""PaddleOCR 3.x adapter for Vietnamese document OCR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class PaddleOCREngine:
    """Normalise PaddleOCR 3.x results into reading-order plain text.

    ``lang='vi'`` selects PaddleOCR's Vietnamese-capable multilingual model.
    The old hard-coded ``ch_*`` recognition model is intentionally not reused.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._ocr = None

    def _init_ocr(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise ImportError(
                "Chưa cài PaddleOCR 3.x hoặc PaddlePaddle 3.x. "
                "Chạy: python -m pip install paddlepaddle==3.2.0 paddleocr==3.3.3"
            ) from error

        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        try:
            self._ocr = PaddleOCR(
                # PP-OCRv5 is the latest OCR-version selector exposed by
                # PaddleOCR 3.3.x.  Its multilingual recognition model
                # includes Vietnamese (lang="vi").
                lang="vi",
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as error:
            raise RuntimeError(f"Không thể khởi tạo PaddleOCR 3.x: {error}") from error

    @staticmethod
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

    def ocr_image(self, image_path: str | Path) -> str:
        self._init_ocr()
        image_path = str(Path(image_path).resolve())
        try:
            results = self._ocr.predict(image_path)
        except Exception as error:
            raise RuntimeError(f"Lỗi khi chạy PaddleOCR 3.x: {error}") from error

        lines: list[tuple[float, float, str]] = []
        for result in results:
            data = self._result_data(result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys")
            if boxes is None:
                boxes = data.get("dt_polys", [])
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 1.0
                if score < self.confidence_threshold or not str(text).strip():
                    continue
                box = boxes[index] if index < len(boxes) else [[0, 0]]
                x, y = float(box[0][0]), float(box[0][1])
                lines.append((y, x, str(text).strip()))

        lines.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(text for _, _, text in lines)
