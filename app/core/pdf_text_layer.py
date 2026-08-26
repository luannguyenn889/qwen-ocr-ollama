"""
Module: pdf_text_layer.py
Nhiệm vụ: Trích xuất lớp văn bản gốc (Text Layer) từ PDF nếu có sẵn để tối ưu hóa thời gian xử lý thay vì dùng OCR Vision.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# pyrefly: ignore [missing-import]
import pymupdf


# Lớp lưu trữ thông tin văn bản trích xuất trực tiếp từ trang PDF
@dataclass(frozen=True)
class NativeTextPage:
    markdown: str          # Văn bản trích xuất định dạng markdown sơ bộ
    character_count: int   # Số lượng ký tự trong văn bản
    word_count: int        # Số lượng từ trong văn bản

    # Thuộc tính kiểm tra xem lớp text layer có đáng tin cậy để sử dụng không
    @property
    def is_usable(self) -> bool:
        """
        Một lớp text layer thực sự cần có đủ nội dung tối thiểu để tin cậy hơn OCR.
        Trả về True nếu số lượng ký tự >= 40 và số từ >= 8.
        """
        return self.character_count >= 40 and self.word_count >= 8


# Hàm dọn dẹp các khối văn bản (bỏ dấu xuống dòng lỗi trong PDF)
def _clean_block(text: str) -> str:
    """
    Giữ nguyên ranh giới đoạn văn đồng thời loại bỏ lỗi tự động xuống dòng trong text PDF gốc.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return " ".join(lines)


# Hàm trích xuất text gốc từ đối tượng trang của PyMuPDF
def extract_native_text(page: pymupdf.Page) -> NativeTextPage:
    """
    Trích xuất các khối văn bản thực tế của PDF theo thứ tự đọc trực quan.

    Hàm này chỉ lấy văn bản, không tự động sinh nhãn ảnh Markdown. Việc xuất ảnh được đảm nhiệm
    bởi một bộ trích xuất riêng biệt có khả năng lưu file ảnh PDF thực tế.
    """
    blocks = page.get_text("blocks", sort=True)
    paragraphs = [_clean_block(block[4]) for block in blocks if block[6] == 0]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    markdown = "\n\n".join(paragraphs)
    plain = re.sub(r"\s+", " ", markdown).strip()
    return NativeTextPage(
        markdown=markdown,
        character_count=len(plain),
        word_count=len(plain.split()),
    )

