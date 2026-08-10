"""Kiểm tra nhanh PaddleOCR có nhận đúng dấu tiếng Việt trên một ảnh.

Chạy:
    .venv\\Scripts\\python.exe test_paddleocr_vi.py duong_dan_anh.png

Đọc kết quả bằng mắt trước khi quyết định dùng PaddleOCR làm OCR đầu vào
cho pipeline. Script này dùng API PaddleOCR 3.x đang khai báo trong dự án.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _data_from_result(result: Any) -> dict[str, Any]:
    """Chuẩn hóa kết quả PaddleOCR 3.x thành dictionary."""
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "res"):
        data = result.res
    elif hasattr(result, "to_dict"):
        data = result.to_dict()
    else:
        data = dict(result)
    return data.get("res", data)


def test_vietnamese_ocr(image_path: Path) -> None:
    from paddleocr import PaddleOCR

    print("Khởi tạo PaddleOCR 3.x với lang='vi' (lần đầu có thể tải model)...")
    ocr = PaddleOCR(
        lang="vi",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(f"Đang OCR ảnh: {image_path.resolve()}")
    results = ocr.predict(str(image_path.resolve()))

    print("\n----- KẾT QUẢ OCR (PaddleOCR, lang=vi) -----\n")
    lines: list[tuple[float, float, str]] = []
    for result in results:
        data = _data_from_result(result)
        texts = data.get("rec_texts", [])
        boxes = data.get("rec_polys")
        if boxes is None:
            boxes = data.get("dt_polys", [])
        for index, text in enumerate(texts):
            box = boxes[index] if index < len(boxes) else [[0, 0]]
            lines.append((float(box[0][1]), float(box[0][0]), str(text)))

    for _, _, text in sorted(lines):
        print(text)

    print("\n----------------------------------------------\n")
    print("Kiểm tra bằng mắt: các dấu ư, ơ, đ, ệ, ữ... có đúng không?")
    print("Nếu sai/mất dấu: không dùng PaddleOCR làm anchor text tuyệt đối;")
    print("hãy để Qwen hybrid đối chiếu trực tiếp ảnh và chỉ sửa khi ảnh rõ.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: .venv\\Scripts\\python.exe test_paddleocr_vi.py duong_dan_anh.png")
        sys.exit(1)

    image = Path(sys.argv[1])
    if not image.is_file():
        print(f"Không tìm thấy ảnh: {image}")
        sys.exit(1)

    test_vietnamese_ocr(image)
