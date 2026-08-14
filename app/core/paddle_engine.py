"""
Module: paddle_engine.py
Nhiệm vụ: Bộ điều hợp (Adapter) cho PaddleOCR 3.x nhằm thực hiện OCR tài liệu tiếng Việt local.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from typing import Any


# Lớp bọc động cơ PaddleOCR để chuẩn hóa kết quả nhận diện thành văn bản thô theo thứ tự đọc
class PaddleOCREngine:
    """
    Chuẩn hóa kết quả của PaddleOCR 3.x thành văn bản thuần túy theo thứ tự đọc.
    
    Sử dụng tham số ``lang='vi'`` để gọi mô hình tiếng Việt chuyên dụng.
    """

    # Hàm khởi tạo động cơ OCR
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._ocr = None

    # Hàm khởi tạo nội bộ động cơ PaddleOCR (Lazy load khi cần)
    def _init_ocr(self) -> None:
        """
        Khởi tạo thực thể PaddleOCR khi sử dụng lần đầu (Lazy loading).
        Thiết lập các biến môi trường để tối ưu hóa CPU nếu cần.
        """
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise ImportError(
                "Chưa cài PaddleOCR 3.x hoặc PaddlePaddle 3.x. "
                "Chạy: python -m pip install paddlepaddle==3.2.0 paddleocr==3.3.3"
            ) from error

        # Thiết lập tối ưu hóa đa luồng cho MKL / MKLDNN
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        try:
            self._ocr = PaddleOCR(
                # PP-OCRv5 là phiên bản OCR mới nhất được hỗ trợ bởi PaddleOCR 3.3.x.
                # Tích hợp sẵn mô hình đa ngôn ngữ hỗ trợ tiếng Việt (lang="vi").
                lang="vi",
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as error:
            raise RuntimeError(f"Không thể khởi tạo PaddleOCR 3.x: {error}") from error

    # Hàm tĩnh phân tách dữ liệu kết quả từ các đối tượng PaddleOCR
    @staticmethod
    def _result_data(result: Any) -> dict[str, Any]:
        """
        Trích xuất dữ liệu thô từ kết quả dự đoán của PaddleOCR để tương thích giữa các phiên bản.
        """
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "res"):
            data = result.res
        elif hasattr(result, "to_dict"):
            data = result.to_dict()
        else:
            data = dict(result)
        return data.get("res", data)

    # Hàm chính thực hiện nhận diện chữ viết trên hình ảnh
    def ocr_image(self, image_path: str | Path) -> str:
        """
        Thực hiện OCR hình ảnh được cung cấp bằng mô hình PaddleOCR.
        Sắp xếp văn bản nhận diện được theo thứ tự từ trên xuống dưới, từ trái sang phải.
        """
        self._init_ocr()
        image_path = str(Path(image_path).resolve())
        try:
            results = self._ocr.predict(image_path)
        except Exception as error:
            raise RuntimeError(f"Lỗi khi chạy PaddleOCR 3.x: {error}") from error

        items: list[tuple[float, float, float, float, str]] = []
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
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                x = min(xs)
                center_y = (min(ys) + max(ys)) / 2
                height = max(max(ys) - min(ys), 1.0)
                items.append((center_y, x, min(ys), max(ys), str(text).strip()))

        if not items:
            return ""

        # Cluster boxes whose vertical centres differ only by scan/detection
        # jitter, then sort fragments left-to-right inside each visual line.
        typical_height = median(item[3] - item[2] for item in items)
        base_threshold = max(3.0, typical_height * 0.5)
        rows: list[list[tuple[float, float, float, float, str]]] = []
        for item in sorted(items, key=lambda value: value[0]):
            if not rows:
                rows.append([item])
                continue
            row = rows[-1]
            row_center = sum(value[0] for value in row) / len(row)
            row_top = median(value[2] for value in row)
            row_bottom = median(value[3] for value in row)
            overlap = max(0.0, min(item[3], row_bottom) - max(item[2], row_top))
            min_height = max(1.0, min(item[3] - item[2], row_bottom - row_top))
            same_line = overlap / min_height >= 0.35 or abs(item[0] - row_center) <= base_threshold * 0.65
            if same_line:
                row.append(item)
            else:
                rows.append([item])

        return "\n".join(
            " ".join(item[4] for item in sorted(row, key=lambda value: value[1]))
            for row in rows
        )
